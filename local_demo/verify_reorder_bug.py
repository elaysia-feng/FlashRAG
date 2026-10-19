"""验证 MultiRetrieverRouter.reorder 的分组前提是否被 as_completed 破坏。

reorder 用 result_list[q_idx + r_idx * query_num] 做位置化分组，
这要求 result_list 是「检索器主序」（retriever_list 顺序）。
但 _search_or_batch_search 用 as_completed 收集，是「完成顺序」。
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flashrag.retriever.retriever import MultiRetrieverRouter  # noqa: E402


class Fake:
    merge_method = "concat"
    reorder = MultiRetrieverRouter.reorder


def doc(i, source):
    return {"id": i, "source": source}


def collect(retriever_list, delays, batch=True):
    """复刻 _search_or_batch_search 的收集逻辑：as_completed 按完成顺序。

    batch=True 时每个检索器返回嵌套 list（batch_search 的形状），
    reorder 才会真正走位置化重排分支。
    """
    result_list = []

    def work(name, delay):
        import time

        time.sleep(delay)
        if batch:
            return [[doc(f"{name}-q{q}-d0", name)] for q in range(2)]
        return [doc(f"{name}-q{q}-d0", name) for q in range(2)]

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_to_name = {
            ex.submit(work, name, delays[name]): name for name in retriever_list
        }
        for fut in as_completed(fut_to_name):
            result_list.extend(fut.result())
    return result_list


f = Fake()

print("=" * 70)
print("场景 A：bm25 快、e5 慢（GPU 检索的典型情况）")
print("=" * 70)
names = ["bm25", "e5", "contriever"]
delays = {"bm25": 0.05, "e5": 0.45, "contriever": 0.20}

collected = collect(names, delays)
print(f"  期望顺序（retriever_list）: {names}")
print(f"  实际收集顺序              : {[g[0]['source'] for g in collected]}")

reordered, _ = f.reorder(collected, [], names)
print()
print("  reorder 后的分组:")
for q, group in enumerate(reordered):
    print(f"    query{q}: {[d['source'] for d in group]}")

print()
print("  >>> 若每组不是 ['bm25','e5','contriever'] 各一条，说明分组被错位")
ok = all([d["source"] for d in g] == names for g in reordered)
print(f"  >>> 分组正确? {'是' if ok else '否 —— BUG 确认'}")

print()
print("=" * 70)
print("场景 B：直接按 retriever_list 顺序收集（对照组，理论上正确）")
print("=" * 70)
ordered = [[doc(f"{n}-q{q}-d0", n)] for n in names for q in range(2)]
reordered2, _ = f.reorder(ordered, [], names)
ok2 = all([d["source"] for d in g] == names for g in reordered2)
print(f"  >>> 分组正确? {'是（对照组成立，说明 reorder 本身没问题）' if ok2 else '否'}")
print(f"  对照组分法: {[[d['source'] for d in g] for g in reordered2]}")
