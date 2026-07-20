# coding=utf-8

import logging
import pickle
import re
from dataclasses import dataclass
from typing import List, Optional, Union
import json

import torch


logger = logging.getLogger(__name__)


def pre_process_text(text: str) -> str:
    text = re.sub(r"<[a-zA-Z/].*?>", "", text)
    text = re.sub(r"http[\x00-\x1f\x21-\xff]+", "", text)
    text = re.sub(r"www\.[\x00-\x1f\x21-\xff]+", "", text)
    text = re.sub(r"&nbsp;|&quot;|\u200b|\u3000", "", text)
    return text


def normalize_labels(label):
    """
    将标签统一成列表，避免字符串标签在 for 循环中被按单个字符拆开。
    """
    if label is None:
        return []
    if isinstance(label, str):
        return [label]
    if isinstance(label, (list, tuple, set)):
        return list(label)
    return [str(label)]
    

@dataclass
class InputExampleForBERT:
    text_a: str
    label: Optional[Union[str, List[str]]] = None


@dataclass(frozen=True)
class InputFeatureForBERT:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    label: Optional[Union[int, List[int]]] = None


class CustomDatasetForBERT(torch.utils.data.Dataset):
    def __init__(self, features,):
        """
        Args:
            features: list of InputFeatureForBERT.
        """
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature: InputFeatureForBERT = self.features[index]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        output = (input_ids, attention_mask)
        if feature.label is not None:
            label = torch.tensor(feature.label, dtype=torch.long)
            output += (label,)
            
        return output

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj


class DataProcessorForBERT:
    def __init__(
        self,
        label_map,
        truncation = 'first-last-256',
        examples=None
    ):
        """
        Args:
            label_map: dict[string, int]. 原始标签(key)与标签序号(value)的对应关系（确保原始标签为str类型）
            examples: (Optional) list of InputExampleForBERT.
        """
        self.label_map = label_map
        self.num_labels = len(label_map)
        self.examples = [] if examples is None else examples
        self.truncation = truncation

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
        
    def _iter_examples_from_json_or_jsonl(self, input_file):
        """
        兼容读取 .json 和 .jsonl。

        .jsonl：逐行读取，每行一个字典，避免 json.load 一次性读入整个大文件导致 MemoryError。
        .json ：保持原来的 JSON 列表读取方式，适合较小的验证集。
        """
        if input_file.lower().endswith(".jsonl"):
            with open(input_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        row = json.loads(line)
                    except Exception as e:
                        logger.warning("JSONL解析失败，文件: %s，行号: %s，错误: %s", input_file, line_no, repr(e))
                        continue

                    if not isinstance(row, dict):
                        logger.warning("JSONL格式错误，文件: %s，行号: %s，该行不是dict，而是: %s", input_file, line_no, type(row))
                        continue

                    yield row
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                df = json.load(f)

            if not isinstance(df, list):
                raise ValueError("{} 根对象不是JSON列表，而是 {}".format(input_file, type(df)))

            for row in df:
                if isinstance(row, dict):
                    yield row

    def add_examples_from_json(
        self,
        input_file,
        overwrite=True,
        ):
        """
        Args:
            input_file: 数据存放地址，支持 .json 列表文件和 .jsonl 文件。
        Returns:
            self.examples
        """
        if overwrite:
            self.examples = []

        valid_count = 0
        skip_count = 0

        for row_idx, row in enumerate(self._iter_examples_from_json_or_jsonl(input_file)):
            if "content" not in row:
                logger.warning("缺少content字段，文件: %s，样本序号: %s", input_file, row_idx)
                skip_count += 1
                continue

            text_a = pre_process_text(str(row["content"]))
            if "label" in row:
                label = row["label"]
            else:
                logger.warning("缺少label字段，文件: %s，样本序号: %s", input_file, row_idx)
                skip_count += 1
                continue

            self.examples.append(InputExampleForBERT(text_a=text_a, label=label))
            valid_count += 1

            if valid_count % 100000 == 0:
                logger.info("已读取有效样本 %s 条", valid_count)

        logger.info("读取原始数据完成，有效样本 %s 条，跳过 %s 条", valid_count, skip_count)

        return self.examples

    def get_feature(
        self,
        tokenizer,
        max_length,
        skip_empty_input,
        loss_type
        ):
        """将样本转化为模型输入
        Args:
            tokenizer:
            max_length: int.
            skip_empty_input: bool.
        Returns:
        """
        features = []
        tokenize_batch_size = 128
        for start in range(0, len(self.examples), tokenize_batch_size):
            batch_examples = self.examples[start:start + tokenize_batch_size]
            valid_examples = []
            for offset, example in enumerate(batch_examples):
                i = start + offset
                if i % 10000 == 0:
                    logger.info("Tokenizing example {}".format(i))

                if skip_empty_input and example.text_a == "":
                    continue
                valid_examples.append((i, example))

            if not valid_examples:
                continue

            tokens = tokenizer(
                text=[example.text_a for _, example in valid_examples],
                max_length=max_length,
                padding="max_length",
                add_special_tokens=True
            )

            for token_idx, (i, example) in enumerate(valid_examples):
                # 文本前后256个token截断。
                if len(tokens['input_ids'][token_idx]) > 512:
                    input_ids = tokens['input_ids'][token_idx][:256] + tokens['input_ids'][token_idx][-256:]
                    attention_mask = tokens['attention_mask'][token_idx][:512]
                else:
                    input_ids = tokens['input_ids'][token_idx]
                    attention_mask = tokens['attention_mask'][token_idx]

                label = None

                labels = normalize_labels(example.label)
                if not labels:
                    logger.warning("样本标签为空，样本序号: %s", i)
                    continue

                if loss_type in {"CE", "Focal"}:
                    lb = labels[0]
                    if lb not in self.label_map:
                        logger.warning("未知标签: %s，样本序号: %s", lb, i)
                        continue
                    label = self.label_map[lb]
                # multi-hot if not single label
                else:
                    label = [0] * self.num_labels
                    matched_count = 0
                    for lb in labels:
                        if lb in self.label_map:
                            label[self.label_map[lb]] = 1
                            matched_count += 1
                    if matched_count == 0:
                        logger.warning("样本标签未命中标签表，标签: %s，样本序号: %s", labels, i)
                        continue
                features.append(InputFeatureForBERT(input_ids=input_ids, attention_mask=attention_mask, label=label))

        logger.info("create {} input features in total.".format(len(features)))
        
        return CustomDatasetForBERT(features)
