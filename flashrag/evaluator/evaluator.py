import os
from flashrag.evaluator.metrics import BaseMetric


class Evaluator:
    """Evaluator is used to summarize the results of all metrics."""
    # 统一管理指标注册、数据集评分和结果持久化。

    def __init__(self, config):
        self.config = config
        self.save_dir = config["save_dir"]

        self.save_metric_flag = config["save_metric_score"]
        self.save_data_flag = config["save_intermediate_data"]
        self.metrics = [metric.lower() for metric in self.config["metrics"]]

        self.avaliable_metrics = self._collect_metrics()

        self.metric_class = {}  
        for metric in self.metrics:
            if metric in self.avaliable_metrics:
                self.metric_class[metric] = self.avaliable_metrics[metric](self.config)
            else:
                print(f"{metric} has not been implemented!")
                raise NotImplementedError

    def _collect_metrics(self):
        """Collect all classes based on ```BaseMetric```."""
        # 递归发现 BaseMetric 子类，以 metric_name 建立指标注册表。

        def find_descendants(base_class, subclasses=None):
            if subclasses is None:
                subclasses = set()

            direct_subclasses = base_class.__subclasses__()
            for subclass in direct_subclasses:
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    find_descendants(subclass, subclasses)
            return subclasses

        avaliable_metrics = {}
        for cls in find_descendants(BaseMetric):
            metric_name = cls.metric_name
            avaliable_metrics[metric_name] = cls
        return avaliable_metrics

    def evaluate(self, data):
        """Calculate all metric indicators and summarize them."""
        # 在生成完成后统一计算评估指标并整理结果。

        result_dict = {}
        for metric in self.metrics:
            try:
                metric_result, metric_scores = self.metric_class[metric].calculate_metric(data)
                result_dict.update(metric_result)

                for metric_score, item in zip(metric_scores, data):
                    item.update_evaluation_score(metric, metric_score)
            except Exception as e:
                print(f"Error in {metric}: {e}")
                continue

        if self.save_metric_flag:
            self.save_metric_score(result_dict)

        if self.save_data_flag:
            self.save_data(data)

        return result_dict

    def save_metric_score(self, result_dict, file_name="metric_score.txt"):
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")

    def save_data(self, data, file_name="intermediate_data.json"):
        """Save the evaluated data, including the raw data and the score of each data
        sample on each metric."""
        # 保存含预测、中间结果及单样本指标的数据集。

        save_path = os.path.join(self.save_dir, file_name)

        data.save(save_path)
