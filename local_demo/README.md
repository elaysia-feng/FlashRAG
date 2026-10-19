# FlashRAG 本地运行说明（Windows / 国内网络）

本目录是在本机把 FlashRAG 跑起来的过程记录与可复现脚本，
**不修改项目原有代码**，只是补齐依赖、绕开网络限制、并提供一个最小可运行示例。

## 1. 环境

| 项 | 值 |
|---|---|
| 解释器 | CPython 3.12.13（uv 管理，`Astral/CPython3.12.13`） |
| 虚拟环境 | 项目根 `.venv/`（即已有的 uv 环境，未重建） |
| 包管理器 | uv 0.11.7 |
| GPU | NVIDIA RTX 4060 Laptop 8GB |
| torch | 2.11.0+cu128（**CUDA 可用**，`torch.cuda.is_available() == True`） |

完整版本清单见 `requirements-resolved.txt`（214 个包，`uv pip freeze` 导出）。

## 2. 网络限制与应对

本机实测：`pypi.org`、`huggingface.co`、`download.pytorch.org` **全部连接超时**，
以下镜像可达：

| 用途 | 不可达 | 改用 |
|---|---|---|
| Python 包 | `pypi.org` | 清华源 `https://pypi.tuna.tsinghua.edu.cn/simple` |
| PyTorch CUDA 轮子 | `download.pytorch.org` | 阿里云 `https://mirrors.aliyun.com/pytorch-wheels/cu128/` |
| 模型权重 | `huggingface.co`、`hf-mirror.com` | ModelScope（`modelscope` SDK） |

注意：阿里云的 pytorch-wheels 是**平铺目录**而非 PEP 503 索引，
必须用 `--find-links` 而不是 `--index-url`：

```powershell
$env:VIRTUAL_ENV="$PWD\.venv"
$env:UV_LINK_MODE="copy"
uv pip install "torch==2.11.0+cu128" `
   --find-links "https://mirrors.aliyun.com/pytorch-wheels/cu128/" `
   --index-url  "https://pypi.tuna.tsinghua.edu.cn/simple"
```

首次安装依赖同样要带镜像：

```powershell
$env:VIRTUAL_ENV="$PWD\.venv"
$env:UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
$env:UV_LINK_MODE="copy"
uv pip install -r requirements.txt
```

## 3. requirements.txt 缺的依赖

`requirements.txt` 只覆盖 `core`，但代码实际 import 了这些没写进清单的包，
**按原样装完仍然 import 失败**。逐个补装：

| 包 | 被谁 import | 不装的后果 |
|---|---|---|
| `dill` | `flashrag.utils.utils` | `import flashrag.utils` 直接 ImportError |
| `faiss-cpu` | `flashrag.retriever.retriever` | 稠密检索不可用 |
| `sentence-transformers` | `flashrag.retriever.encoder` | `import flashrag.retriever` 失败 |
| `PyStemmer` | `flashrag.retriever.retriever`（BM25 后端） | bm25s 后端不可用 |
| `termcolor` | `flashrag.pipeline.*` | `import flashrag.pipeline` 失败 |
| `pyserini` | BM25 的 pyserini 后端（函数内延迟 import） | 该后端不可用 |
| `modelscope` | 仅本目录下载脚本使用 | 无法取模型 |

```powershell
uv pip install dill faiss-cpu sentence-transformers PyStemmer termcolor pyserini modelscope
```

### JAVA_HOME

`pyserini` 需要 JDK。本机 JDK 在 `E:\develop\java\jdk-21`，
已写入当前用户的 `JAVA_HOME` 环境变量（原先未设置，pyserini 报 `Unable to find JAVA_HOME`）。
新开终端自动生效；当前会话可用 `$env:JAVA_HOME="E:\develop\java\jdk-21"` 临时设置。

## 4. 已知不可用项：SPLADE / seismic

`flashrag/retriever/retriever.py:1011` 需要 `from seismic import SeismicIndex`。
PyPI 上的 `seismic` 是同名的**地震学**库（依赖 obspy/mpi4py），不提供 `SeismicIndex`，
因此该 SPLADE 检索路径在 Windows 上无法工作。
真正的实现来自 Seismic 论文的独立仓库，且面向 CUDA/Linux。

影响范围仅限 `retrieval_method: "splade"`；
稠密检索（dense）与 BM25 均不受影响。

## 5. 运行

### 5.1 下载模型

```powershell
uv run --no-sync python local_demo/fetch_models.py --with-generator
```

| 模型 | ModelScope ID | 大小 | 落地目录 |
|---|---|---|---|
| 检索 | `BAAI/bge-small-zh-v1.5` | ~95 MB | `models/bge-small-zh-v1.5` |
| 生成 | `Qwen/Qwen2.5-1.5B-Instruct` | ~3.1 GB | `models/Qwen2.5-1.5B-Instruct` |

选型理由：8GB 显存放不下官方 demo 的 `Qwen1.5-14B`（约 28GB），
1.5B 约占 3GB，与检索模型同驻仍有余量。

### 5.2 跑示例

```powershell
# 完整 RAG（建索引 -> 召回 -> 生成）
uv run --no-sync python local_demo/run_demo.py

# 只看检索，不加载生成模型
uv run --no-sync python local_demo/run_demo.py --skip-generate
```

语料是 `datasets/local_demo_corpus.jsonl`（12 条中文 AI/RAG 知识，
格式为 FlashRAG 要求的 `{"id", "contents"}` JSONL）。

### 5.3 跑单元测试

```powershell
uv run --no-sync python -m unittest tests.test_retrieval_precision -v
uv run --no-sync python -m unittest tests.test_empty_prediction_metrics -v
```

这两个套件是纯逻辑测试，不下载模型，可用于快速验证代码改动。

## 6. 官方 demo 为什么跑不了

`examples/quick_start/demo_zh.py` 硬编码了作者服务器上的路径与模型：

- `generator_model: qwen-14B` → `Qwen/Qwen1.5-14B-Chat`，8GB 显存放不下
- `corpus_path: /data00/jiajie_jin/rd_corpus.jsonl` → 本机不存在
- `index_path: /data00/jiajie_jin/flashrag_indexes/...` → 本机不存在

所以用 `local_demo/run_demo.py` 走同一套 FlashRAG 组件（`Config` /
`get_retriever` / `Index_Builder` / `PromptTemplate` / `get_generator`），
只是换成能装下的模型与本地语料。

## 7. 目录说明

| 文件 | 说明 |
|---|---|
| `run_demo.py` | 最小 RAG 示例 |
| `fetch_models.py` | 经 ModelScope 下载模型 |
| `env_probe.py` | 逐层检查 `flashrag.*` 各子包能否 import |
| `requirements-resolved.txt` | `uv pip freeze` 导出的完整版本清单 |

索引按模型分目录存放：`indexes/local_demo/<模型名>/dense_Flat.index`，
换检索模型不会误用上一次的旧索引。
