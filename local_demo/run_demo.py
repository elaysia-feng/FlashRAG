"""FlashRAG 本地最小可运行示例。

流程：构建稠密检索索引 -> 召回 -> 组装 RAG 提示词 -> 生成回答。
模型默认从项目内 models/ 目录读取（国内网络走 ModelScope 下载，见 fetch_models.py）。

用法：
    uv run --no-sync python local_demo/run_demo.py
    uv run --no-sync python local_demo/run_demo.py --skip-generate
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CORPUS_PATH = os.path.join(ROOT, "datasets", "local_demo_corpus.jsonl")
RETRIEVER_MODEL = os.path.join(ROOT, "models", "bge-small-zh-v1.5")
GENERATOR_MODEL = os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct")

QUERIES = [
    "QLoRA 是什么，它为什么能在单卡上微调大模型？",
    "BM25 里的 k1 和 b 分别控制什么？",
    "vLLM 的 PagedAttention 解决了什么问题？",
]


def index_dir_for(model_path):
    """索引目录按模型名隔离，切换检索模型不会误用上一次的旧索引。"""
    return os.path.join(ROOT, "indexes", "local_demo", os.path.basename(model_path.rstrip("/\\")))


def build_index(model_path, pooling_method):
    """构建 FAISS 索引。"""
    from flashrag.retriever.index_builder import Index_Builder

    save_dir = index_dir_for(model_path)
    print(f"[1/4] 构建稠密索引 -> {save_dir}")
    Index_Builder(
        retrieval_method="dense",
        model_path=model_path,
        corpus_path=CORPUS_PATH,
        save_dir=save_dir,
        max_length=512,
        batch_size=8,
        use_fp16=True,
        pooling_method=pooling_method,
    ).build_index()


def resolve_index_path(model_path):
    """索引文件名由 Index_Builder 决定（dense_Flat.index），这里自动定位。"""
    index_dir = index_dir_for(model_path)
    if not os.path.isdir(index_dir):
        return None
    for name in sorted(os.listdir(index_dir)):
        if name.endswith(".index"):
            return os.path.join(index_dir, name)
    return None


def build_config(model_path, pooling_method, index_path):
    from flashrag.config import Config

    return Config(
        config_dict={
            "save_note": "local_demo",
            "gpu_id": "0",
            # 检索相关配置
            "retrieval_method": "dense",
            "retrieval_model_path": model_path,
            "index_path": index_path,
            "corpus_path": CORPUS_PATH,
            "retrieval_topk": 3,
            "retrieval_use_fp16": True,
            "retrieval_pooling_method": pooling_method,
            # BGE 中文模型要求给查询加前缀，否则召回质量明显下降。
            # FlashRAG 依据 retrieval_method 的子串自动推断，这里显式指定更可靠。
            "instruction": "为这个句子生成表示以用于检索相关文章：",
            "use_retrieval_cache": False,
            "save_retrieval_cache": False,
            # 生成相关配置
            "framework": "hf",
            "generator_model": "qwen",
            "generator_model_path": GENERATOR_MODEL,
            "generator_max_input_len": 2048,
            "generator_batch_size": 1,
            "generation_params": {"max_tokens": 256, "do_sample": False},
        }
    )


def run_retrieval(config):
    from flashrag.utils import get_retriever

    print("\n[2/4] 加载检索器并召回")
    retriever = get_retriever(config)
    results = [retriever.search(q, num=3) for q in QUERIES]
    for query, docs in zip(QUERIES, results):
        print(f"\n  Q: {query}")
        for doc in docs:
            print(f"     - [{doc.get('id')}] {doc.get('contents', '')[:56]}...")
    return results


def run_generation(config, results):
    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_generator

    system_prompt = (
        "你是一个友好的人工智能助手。请根据下面提供的参考资料回答用户问题，"
        "只使用资料中的信息，不要编造。\n\n参考资料：\n{reference}"
    )
    prompt_template = PromptTemplate(
        config, system_prompt=system_prompt, user_prompt="{question}"
    )

    print("\n[3/4] 加载生成模型")
    generator = get_generator(config)

    prompts = [
        prompt_template.get_string(question=q, retrieval_result=d)
        for q, d in zip(QUERIES, results)
    ]

    print("[4/4] 批量生成回答\n")
    answers = generator.generate(prompts)
    for query, answer in zip(QUERIES, answers):
        print(f"  Q: {query}")
        print(f"  A: {answer.strip()}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generate", action="store_true", help="只跑检索")
    parser.add_argument("--skip-build", action="store_true", help="跳过索引构建")
    parser.add_argument("--retriever", default=RETRIEVER_MODEL)
    parser.add_argument("--pooling", default="cls")
    args = parser.parse_args()

    if not os.path.isdir(args.retriever):
        print(f"[x] 找不到检索模型：{args.retriever}")
        print("    请先执行：uv run --no-sync python local_demo/fetch_models.py")
        return 1

    if not args.skip_build:
        build_index(args.retriever, args.pooling)

    index_path = resolve_index_path(args.retriever)
    if index_path is None:
        print(f"[x] 在 {index_dir_for(args.retriever)} 下找不到 *.index 索引文件")
        return 1

    config = build_config(args.retriever, args.pooling, index_path)
    results = run_retrieval(config)

    if args.skip_generate:
        return 0

    if not os.path.isdir(GENERATOR_MODEL):
        print(f"\n[!] 跳过生成：找不到生成模型 {GENERATOR_MODEL}")
        print("    下载命令：uv run --no-sync python local_demo/fetch_models.py --with-generator")
        return 0

    run_generation(config, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
