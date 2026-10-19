# Source: https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py
# Copyright 2020 SacreBLEU Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from functools import lru_cache


class BaseTokenizer:
    """A base dummy tokenizer to derive from."""
    # 分词器接口基类，供具体 BLEU 分词实现继承。

    def signature(self):
        """
        Returns a signature for the tokenizer.
        :return: signature string
        """
        # 返回分词器的签名信息。
        return "none"

    def __call__(self, line):
        """
        Tokenizes an input line with the tokenizer.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        # 执行当前对象封装的主要操作。
        return line


class TokenizerRegexp(BaseTokenizer):
    def signature(self):
        return "re"

    def __init__(self):
        self._re = [
            # language-dependent part (assuming Western languages)
            # 与语言相关的处理，这里假设输入为西方语言。
            (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
            # tokenize period and comma unless preceded by a digit
            # 对前面不是数字的句点和逗号进行分词。
            (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
            # tokenize period and comma unless followed by a digit
            # 对后面不是数字的句点和逗号进行分词。
            (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
            # tokenize dash when preceded by a digit
            # 对前面是数字的连字符进行分词。
            (re.compile(r"([0-9])(-)"), r"\1 \2 "),
            # one space only between words
            # 词之间只保留一个空格。
            # NOTE: Doing this in Python (below) is faster
            # 注意：下方直接用 Python 处理的方式更快。
            # (re.compile(r'\s+'), r' '),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        """Common post-processing tokenizer for `13a` and `zh` tokenizers.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        # 执行当前对象封装的主要操作。
        for _re, repl in self._re:
            line = _re.sub(repl, line)

        # no leading or trailing spaces, single space within words
        # 删除首尾空白，并将词间空白合并为一个空格。
        # return ' '.join(line.split())
        # This line is changed with regards to the original tokenizer (seen above) to return individual words
        # 与上方原始分词器不同，此处返回单独的词列表。
        return line.split()


class Tokenizer13a(BaseTokenizer):
    def signature(self):
        return "13a"

    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        """Tokenizes an input line using a relatively minimal tokenization
        that is however equivalent to mteval-v13a, used by WMT.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        # 执行当前对象封装的主要操作。

        # language-independent part:
        # 与具体语言无关的处理。
        line = line.replace("<skipped>", "")
        line = line.replace("-\n", "")
        line = line.replace("\n", " ")

        if "&" in line:
            line = line.replace("&quot;", '"')
            line = line.replace("&amp;", "&")
            line = line.replace("&lt;", "<")
            line = line.replace("&gt;", ">")

        return self._post_tokenizer(f" {line} ")


# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    # 统计指定阶数以内的连续 token 片段，供 BLEU 计算匹配数量。
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    # 结合 n-gram 精确率和长度惩罚计算 BLEU，并返回相关统计量。
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    elif ratio == 0.0:
        # An empty translation gets no credit, as in sacrebleu.
        # 与 sacrebleu 保持一致：空翻译不计分。
        bp = 0.0
    else:
        bp = math.exp(1 - 1.0 / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)
