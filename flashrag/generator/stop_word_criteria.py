"""
Created by Nestor Demeure.
This software is released under the Apache License 2.0.
"""

from typing import List
import torch
from transformers import StoppingCriteria, AutoTokenizer


class StopWordCriteria(StoppingCriteria):
    """
    A stopping criteria that halts the text generation process if any specified stop word is encountered.

    Inspired by https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/9
    And: https://github.com/outlines-dev/outlines/blob/main/outlines/generate/api.py
    """
    # 在生成文本遇到指定停止词时控制生成结束。

    def __init__(self, tokenizer: AutoTokenizer, prompts: List[str], stop_words: List[str] = [], check_every: int = 1):
        """
        Initializes the StopWordCriteria with the necessary parameters for checking stop words during text generation.

        Parameters:
            tokenizer (AutoTokenizer): The tokenizer for encoding prompts and stop words.
            prompts (List[str]): Initial prompts used for generation, needed to determine where generated text begins.
            stop_words (List[str]): Words that trigger the stopping of generation when detected.
            check_every (int): Frequency of checking for stop words in the token stream (a performance optimization, use 1 to cut it out).
        """
        # 初始化当前组件的配置、资源与运行状态。
        super().__init__()
        self.tokenizer = tokenizer
        self.input_sizes = [self.tokenizer.encode(prompt, return_tensors="pt").size(-1) for prompt in prompts]
        self.stop_words = stop_words
        self.max_stop_word_size = max(
            (self.tokenizer.encode(word, return_tensors="pt").size(-1) for word in stop_words), default=0
        )
        self.check_every = check_every

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Determines whether to stop generation based on the presence of stop words.

        Stops if a stop word is found in *all* batch elements *and* the sequence length is a multiple of `check_every`.
        Note: Delay in stopping may occur if `check_every > 1`.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            scores (torch.FloatTensor): Generation scores for each token. Not used here.

        Returns:
            bool: True to stop generation, False to continue.
        """
        # 执行当前对象封装的主要操作。
        batch_size, seq_len = input_ids.shape

        # Skip check if no stop words are defined or it is not yet time to check
        # 未设置停止词或尚未到检查时机时跳过检查。
        if (len(self.stop_words) == 0) or (seq_len % self.check_every != 0):
            return False

        for i in range(batch_size):
            # Calculate starting index for new tokens
            # 计算新生成 token 的起始位置。
            prompt_size = self.input_sizes[i]
            max_new_tokens = (2 * self.max_stop_word_size) + self.check_every
            latest_tokens = input_ids[i, prompt_size:][-max_new_tokens:]

            # Check for stop words in the decoded text
            # 在解码文本中检查停止词。
            if not any(
                word in self.tokenizer.decode(latest_tokens, skip_special_tokens=True) for word in self.stop_words
            ):
                return False  # Continue generation if any batch item lacks stop words
                # 批次中只要还有样本未出现停止词，就继续生成。

        return True  # Stop generation if all conditions are met
        # 所有停止条件满足后结束生成。

    def extract_answers(self, input_ids: torch.LongTensor, strip_stopword: bool = True) -> List[str]:
        """
        Extracts generated answers by removing prompts and optionally stopping at the first stop word.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            strip_stopword (bool): Determines whether the stop word is removed from the output.

        Returns:
            List[str]: Extracted answers, with or without stop words.
        """
        # 去掉输入提示对应的 token，并按开关截断或保留首次出现的停止词。
        batch_size, _ = input_ids.shape
        result = []

        for i in range(batch_size):
            # Decode generated tokens to text, excluding the prompt
            # 仅解码新生成的 token，不包含输入提示词。
            prompt_size = self.input_sizes[i]
            answer_tokens = input_ids[i, prompt_size:]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # Find the first occurrence of any stop word
            # 查找任意停止词首次出现的位置。
            lower_stop_index = len(answer_text)  # Default to end of text
            # 默认截取到文本末尾。
            for word in self.stop_words:
                stop_index = answer_text.find(word)
                if stop_index != -1:
                    # Adjust stop index based on whether we're stripping the stop word
                    # 根据是否移除停止词调整截断位置。
                    stop_index += 0 if strip_stopword else len(word)
                    lower_stop_index = min(stop_index, lower_stop_index)

            # Cut the text at the first stop word found (if any)
            # 在找到的第一个停止词处截断文本。
            answer_text = answer_text[:lower_stop_index]
            result.append(answer_text)

        return result
