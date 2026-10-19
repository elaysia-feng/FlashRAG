"""环境探针：逐层验证 flashrag 各子包能否导入，并报告失败原因。"""

import os
import sys
import traceback

# 本脚本位于 local_demo/，需把项目根加入 sys.path 才能 import flashrag。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

print(f"python  : {sys.version.split()[0]}")
print(f"prefix  : {sys.prefix}")
print(f"root    : {ROOT}")
print("")

checks = [
    ("torch + CUDA", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"),
    ("numpy", "import numpy; print('numpy', numpy.__version__)"),
    ("faiss", "import faiss; print('faiss', getattr(faiss, '__version__', 'ok'))"),
    ("bm25s", "import bm25s; print('bm25s ok')"),
    ("Stemmer", "import Stemmer; print('PyStemmer ok')"),
    ("pyserini", "from pyserini.search.lucene import LuceneSearcher; print('pyserini ok')"),
    ("seismic", "from seismic import SeismicIndex; print('seismic ok')"),
    ("sentence_transformers", "import sentence_transformers; print('ST', sentence_transformers.__version__)"),
    ("flashrag.evaluator.metrics", "from flashrag.evaluator.metrics import Rouge_1; print('metrics ok')"),
    ("flashrag.utils", "from flashrag.utils import get_retriever, get_generator; print('utils ok')"),
    ("flashrag.prompt", "from flashrag.prompt import PromptTemplate; print('prompt ok')"),
    ("flashrag.dataset", "from flashrag.dataset import Dataset; print('dataset ok')"),
    ("flashrag.retriever", "from flashrag.retriever import DenseRetriever, BM25Retriever; print('retriever ok')"),
    ("flashrag.generator", "from flashrag.generator import HFCausalLMGenerator; print('generator ok')"),
    ("flashrag.pipeline", "from flashrag.pipeline import SequentialPipeline; print('pipeline ok')"),
]

failed = 0
for name, stmt in checks:
    try:
        exec(stmt, {})
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        tb = traceback.format_exc().strip().splitlines()[-1]
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        print(f"        {tb[:160]}")

print("")
print(f"result: {len(checks) - failed}/{len(checks)} passed")
sys.exit(1 if failed else 0)
