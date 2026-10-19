"""验证 MultiRetrieverRouter.post_process_result 的 concat 去重分支是否存在真实缺陷。

直接调用被测方法，构造最小输入，不依赖模型或索引。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flashrag.retriever.retriever import MultiRetrieverRouter  # noqa: E402

# 直接取真实方法，绑定到轻量 stub 上，绕开 __init__ 的模型加载。
post_process_result = MultiRetrieverRouter.post_process_result


class Fake:
    """只提供 post_process_result 用到的字段。"""

    def __init__(self, merge_method):
        self.merge_method = merge_method

    post_process_result = post_process_result


def doc(i, source):
    return {"id": i, "contents": f"c{i}", "source": source, "corpus_path": "p", "is_multimodal": False}


print("=" * 68)
print("实验 1：concat + naive(flat dict) + 有重复 id + return_score=True")
print("=" * 68)
f = Fake("concat")
result_list = [doc("d1", "bm25"), doc("d1", "e5"), doc("d2", "bm25")]
score_list = [0.9, 0.8, 0.7]
try:
    r, s = f.post_process_result("q", result_list, score_list, num=5)
    print(f"  返回 result ids : {[d['id'] for d in r]}")
    print(f"  返回 score_list : {s}")
    print(f"  >>> 去重后结果条数 = {len(r)}（期望 2）")
except Exception as exc:
    print(f"  !!! 抛异常: {type(exc).__name__}: {exc}")

print()
print("=" * 68)
print("实验 2：concat + naive + 3 个重复 id（检查跳元素）")
print("=" * 68)
f = Fake("concat")
result_list = [
    doc("d1", "bm25"), doc("d2", "bm25"), doc("d1", "e5"),
    doc("d2", "e5"), doc("d1", "contriever"), doc("d3", "bm25"),
]
try:
    r, s = f.post_process_result("q", result_list, [], num=5)
    ids = [d["id"] for d in r]
    print(f"  返回 ids : {ids}")
    print(f"  >>> 期望 ['d1','d2','d3']，实际 {ids}")
    print(f"  >>> 残留重复项: {'有（BUG）' if len(ids) != len(set(ids)) else '无'}")
except Exception as exc:
    print(f"  !!! 抛异常: {type(exc).__name__}: {exc}")

print()
print("=" * 68)
print("实验 3：concat + batch(嵌套 list) + 重复 id + 有分数")
print("=" * 68)
f = Fake("concat")
result_list = [
    [doc("d1", "bm25"), doc("d2", "bm25"), doc("d1", "e5")],
]
score_list = [[0.9, 0.8, 0.75]]
try:
    r, s = f.post_process_result(["q"], result_list, score_list, num=5)
    ids = [d["id"] for d in r[0]]
    print(f"  返回 ids      : {ids}")
    print(f"  返回 score_list: {s}")
    print(f"  >>> 期望 ids ['d1','d2']，实际 {ids}")
except Exception as exc:
    print(f"  !!! 抛异常: {type(exc).__name__}: {exc}")

