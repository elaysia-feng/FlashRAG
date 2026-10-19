# Source: https://github.com/IntelLabs/fastRAG/blob/main/fastrag/generators/replug.py
# The release is licensed under the Apache License 2.0

import warnings
from typing import List, Optional, Union
import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import (
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleOutput,
)


class REPLUG_Generation(GenerationMixin):
    """Implementing REPLUG-based sampling text generation."""
    # 实现基于 REPLUG 的文本采样生成。

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        # init values
        # 初始化生成参数。
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        # 初始化注意力、隐藏状态和分数的收集容器。
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # 编码器—解码器模型还需取得编码器的注意力权重与隐藏状态。
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        # 跟踪哪些序列已经生成完成。
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # 仅在 synced_gpus 模式下使用。
        # auto-regressive generation
        # 自回归生成循环。
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # GPU 同步模式下，所有 GPU 的序列完成之前仍需继续前向调用。
                # The following logic allows an early break if all peers finished generating their sequence
                # 下方逻辑用于在所有并行进程均生成完成后提前退出。
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                # 当前进程完成时发送 0.0，否则发送 1.0。
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                # 归约后的总和为 0.0，表示所有并行进程都已完成。
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            # 准备模型输入。
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            # 执行前向计算，取得下一 token 的分布。
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
                # 已完成的进程跳过不再需要的后续计算，避免浪费资源。

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            # 对生成分布进行预处理。
            ### REPLUG - document weighting is done via REPLUGLogitsProcessor
            # REPLUG 的文档权重融合由 REPLUGLogitsProcessor 完成。
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            # 按开关保存分数、注意力和隐藏状态。
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            ### REPLUG
            # REPLUG 专用生成逻辑。
            # Sample from the normalized "logits", assuming the REPLUG processor was used!
            # 假定已经应用 REPLUG 处理器，从归一化后的分布中采样。
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Lock same next-token for all examples in batch
            # 使批次中各篇文档的生成路径使用同一个下一 token。
            next_tokens[:] = next_tokens[0]

            # finished sentences should have their next token be a padding token
            # 已完成序列的下一个 token 应使用填充 token。
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # 更新已生成 token、模型输入和长度，供下一步使用。
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            # 某条序列出现结束 token 后，将其标记为完成。
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                # 全部序列完成后停止。
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            # 超过最大长度时停止。
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


class REPLUGLogitsProcessor(LogitsProcessor):
    """
    Merge logits of different docs in one batch.

    Reference: fastRAG
    """
    # 融合一个批次中不同文档对应的 token 分数分布。

    def __init__(self, doc_scores: torch.FloatTensor):
        self.num_docs = doc_scores.shape[0]
        # normalize
        # 归一化处理。
        doc_scores /= doc_scores.sum()
        self.doc_scores = torch.unsqueeze(doc_scores, 1)  # k*1

    def __call__(self, input_ids, scores):
        # doc_score: k*1, scores: k*vocab_size
        # 文档分数形状为 k×1，token 分数形状为 k×词表大小。
        replug_scores = self.doc_scores * scores
        replug_scores = replug_scores.sum(dim=0)  # 1*vocab_size
        # 融合后形状为 1×词表大小。
        replug_scores = torch.tile(replug_scores, (self.num_docs, 1))  # k*vocab_size
        # 扩展后的形状为 k×词表大小。
        return replug_scores


def load_replug_model(name_or_path):
    class HF_REPLUG:
        "Creates a HF model that inherits from REPLUG_Generation class"
        # 创建继承 REPLUG 生成能力的 Hugging Face 模型类。

        def __new__(cls, name_or_path, **kwargs):
            return factory(name_or_path).from_pretrained(name_or_path, **kwargs)

    def factory(name_or_path):
        loadedConfig = AutoConfig.from_pretrained(name_or_path)
        try:
            pretrained_class_object = getattr(transformers, loadedConfig.architectures[0])
            if pretrained_class_object not in MODEL_FOR_CAUSAL_LM_MAPPING.values():
                raise ValueError(f"Model {pretrained_class_object} is not used for causal LM generation.")
        except AttributeError:
            raise ValueError("Transformers architecture unknown.")

        class HF(pretrained_class_object, REPLUG_Generation):
            """Wrapper around HuggingFace transformers with REPLUG generation."""
            # 为 Hugging Face 模型包装 REPLUG 生成能力。

            _keys_to_ignore_on_load_unexpected = [r"cls"]
            _tied_weights_keys = ["lm_head.weight"]

        return HF

    return HF_REPLUG(name_or_path)
