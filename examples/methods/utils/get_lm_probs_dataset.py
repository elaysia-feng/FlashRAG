# This file is used for generating LM supervised dataset to finetune retrievers
# 此文件用于生成语言模型监督数据，以微调检索器。
# Implementation details are learned from REPLUG:https://arxiv.org/abs/2301.12652
# 实现细节参考 REPLUG 论文。
import json
import random

import fire
import torch
from tqdm import tqdm

from flashrag.config import Config
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_dataset
from flashrag.utils import get_retriever, get_generator


class LMProbCalculator:
    """
    Clculating the likelihood of the ground truth
    when LM uses every document retrieved from
    top-k retrieved ones in context
    """
    # 计算标准答案的似然概率，供检索器监督数据构建使用。

    def __init__(self, config):
        # Load your own components
        # 加载自行配置的组件。
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)
        self.prompt_template = PromptTemplate(config)

    def run(self, dataset):
        input_query = dataset.question
        answers = [answer[0] if len(answer) == 1 else random.choice(answer) for answer in dataset.golden_answers]
        retrieval_results = self.retriever.batch_search(input_query)
        data_ls = []
        N = len(answers)
        for i in tqdm(range(N)):
            q, res_list, answer = dataset.question[i], retrieval_results[i], answers[i]
            docs = []
            scores = []
            for res in res_list:
                input_prompt = self.prompt_template.get_string(question=q, retrieval_result=[res])
                score = self.calculate_prob(input_prompt, answer)
                scores.append(score)
                docs.append(res["contents"])
            scores = torch.softmax(torch.tensor(scores), dim=-1).tolist()
            data_ls.append({"query": q, "pos": docs, "scores": scores})
        return data_ls

    def calculate_prob(self, prompt, answer):
        _, answer_probs = self.generator.cal_gen_probs(prompt, answer)
        return float(answer_probs.mean())


def main(
    dataset_name="nq",  # qa dataset
    # 问答数据集。
    split="test",  # split
    # 数据划分。
    num=4000,  # number of query-document pairs
    # 查询与文档配对的数量。
    gpu_id="0",
    output="lmsft.jsonl",  # output path
    # 输出路径。
    topk=20,  # number of retrieved documents
    # 召回文档数量。
):
    config_dict = {
        "save_note": "replug_lsr",
        "gpu_id": gpu_id,
        "dataset_name": dataset_name,
        "test_sample_num": num,
        "split": ["test", "dev"],
        "retrieval_topk": topk,
        "framework": "hf",
    }
    config = Config("my_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    lm_prob_calculator = LMProbCalculator(config)
    data_ls = lm_prob_calculator.run(test_data)
    with open(output, "w") as f:
        for data in data_ls:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
