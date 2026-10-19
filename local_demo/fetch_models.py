"""从 ModelScope 下载演示所需模型到项目内 models/ 目录。

国内直连 huggingface.co 超时，因此走 ModelScope 官方源。

用法：
    uv run --no-sync python local_demo/fetch_models.py
    uv run --no-sync python local_demo/fetch_models.py --with-generator
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

RETRIEVER_ID = "BAAI/bge-small-zh-v1.5"
RETRIEVER_BASE_ID = "BAAI/bge-base-zh-v1.5"
GENERATOR_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def download(model_id, local_name):
    from modelscope import snapshot_download

    target = os.path.join(MODEL_DIR, local_name)
    if os.path.isdir(target) and os.listdir(target):
        print(f"已存在，跳过：{target}")
        return target

    print(f"下载 {model_id} -> {target}")
    snapshot_download(model_id, local_dir=target)
    print(f"完成：{target}")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generator", action="store_true", help="一并下载生成模型")
    parser.add_argument(
        "--retriever",
        choices=["small", "base"],
        default="small",
        help="检索模型规格：small(~95MB) 或 base(~400MB，召回更好)",
    )
    args = parser.parse_args()

    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("缺少 modelscope，请先执行：")
        print("  uv pip install modelscope")
        return 1

    os.makedirs(MODEL_DIR, exist_ok=True)
    if args.retriever == "base":
        download(RETRIEVER_BASE_ID, "bge-base-zh-v1.5")
    else:
        download(RETRIEVER_ID, "bge-small-zh-v1.5")
    if args.with_generator:
        download(GENERATOR_ID, "Qwen2.5-1.5B-Instruct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
