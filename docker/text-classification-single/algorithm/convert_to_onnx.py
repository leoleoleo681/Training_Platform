#!/usr/bin/env python3
# coding=utf-8

import argparse
import logging

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


# 如果只是导出 ONNX，一般不会用到 ASL / FocalLoss。
# 这里做兼容，防止环境里没有 loss_function.py 时报错。
try:
    from loss_function import AsymmetricLoss, FocalLoss
except Exception:
    AsymmetricLoss = None
    FocalLoss = None


logger = logging.getLogger(__name__)


class BertForMultiLabel(BertPreTrainedModel):
    """
    与训练代码保持一致的 BertForMultiLabel 模型结构。

    结构包括：
    1. BertModel
    2. first-last-avg 或 cls pooling
    3. dense + tanh
    4. dropout
    5. classifier

    导出 ONNX 时 forward 返回：
    - prob: 每条样本最大类别概率
    - label: 每条样本预测类别 id

    这样可以保持你原第一个文件里的 ONNX 输出格式不变。
    """

    def __init__(
        self,
        config,
        label_distribution=None,
        distribution_gamma=0,
        alpha=2,
        pool_type="first-last-avg"
    ):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 兼容 config 中没有 loss_type 的情况
        self.loss_type = getattr(config, "loss_type", "CE")

        self.pool_type = pool_type
        self.label_distribution = label_distribution
        self.alpha = alpha
        self.distribution_gamma = distribution_gamma

        # 兼容不同 transformers 版本
        if hasattr(self, "post_init"):
            self.post_init()
        else:
            self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        """
        Args:
            input_ids: torch.LongTensor, shape = [batch_size, seq_length]
            attention_mask: torch.LongTensor, shape = [batch_size, seq_length]
            labels: 可选，训练时使用；导出 ONNX 时不传
        """

        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # ======================
        # pooling 方式，与第二个代码保持一致
        # ======================
        if self.pool_type == "cls":
            sequence_output = bert_outputs[0][:, 0]

        elif self.pool_type == "first-last-avg":
            hidden_states = bert_outputs.hidden_states

            # hidden_states[1] 是第一层 Transformer 输出
            first_avg = hidden_states[1].mean(dim=1)

            # hidden_states[-1] 是最后一层 Transformer 输出
            last_avg = hidden_states[-1].mean(dim=1)

            sequence_output = (first_avg + last_avg) / 2

        else:
            raise ValueError(f"不支持的 pool_type: {self.pool_type}")

        pooled_output = self.dense(sequence_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        # ======================
        # 如果传 labels，则保留训练损失逻辑
        # ======================
        if labels is not None:
            if self.loss_type == "BCE":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1, self.num_labels).to(torch.float)
                )
                return loss, logits, pooled_output

            elif self.loss_type == "ASL":
                if AsymmetricLoss is None:
                    raise ImportError("未找到 AsymmetricLoss，请检查 loss_function.py")

                loss_fct = AsymmetricLoss(
                    label_distribution=self.label_distribution,
                    distribution_gamma=2,
                    alpha=self.alpha,
                    gamma_neg=2,
                    gamma_pos=1
                )
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1, self.num_labels).to(torch.float)
                )
                return loss, logits, pooled_output

            elif self.loss_type == "CE":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1)
                )
                return loss, logits, pooled_output

            elif self.loss_type == "Focal":
                if FocalLoss is None:
                    raise ImportError("未找到 FocalLoss，请检查 loss_function.py")

                loss_fct = FocalLoss(
                    label_distribution=self.label_distribution,
                    alpha=self.alpha,
                    gamma=2,
                    distribution_gamma=self.distribution_gamma,
                )
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1, self.num_labels).to(torch.float)
                )
                return loss, logits, pooled_output

        # ======================
        # ONNX 导出 / 推理输出
        # 保持第一个文件原来的输出格式：
        # prob, label
        # ======================
        probs = torch.softmax(logits, dim=-1)
        max_prob, label = torch.max(probs, dim=-1)

        return max_prob, label


def convert_pt_to_onnx(model_dir, device, seq_max_length, onnx_file, pool_type="first-last-avg"):
    """
    Args:
        model_dir: pytorch 模型文件夹路径
        device: cpu 或 cuda
        seq_max_length: 最大输入长度
        onnx_file: onnx 模型导出路径
        pool_type: 需要和训练时保持一致，默认 first-last-avg
    """

    device = torch.device(device)

    dummy_input_ids = torch.ones(
        (1, seq_max_length),
        dtype=torch.long
    ).to(device)

    dummy_attn_mask = torch.ones(
        (1, seq_max_length),
        dtype=torch.long
    ).to(device)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["prob", "label"]

    dynamic_axes = {
        "input_ids": {
            0: "batch_size",
            1: "max_seq_length"
        },
        "attention_mask": {
            0: "batch_size",
            1: "max_seq_length"
        },
        "prob": {
            0: "batch_size"
        },
        "label": {
            0: "batch_size"
        }
    }

    # ======================
    # 使用新的 BertForMultiLabel 加载模型
    # ======================
    model = BertForMultiLabel.from_pretrained(
        model_dir,
        pool_type=pool_type
    )

    model.eval()
    model.to(device)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attn_mask),
        onnx_file,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=11,
        dynamic_axes=dynamic_axes
    )

    logger.info(f"成功将模型导出到文件 {onnx_file}")

    # ======================
    # 验证 PyTorch 和 ONNXRuntime 输出是否一致
    # ======================
    logger.info("验证 pytorch 和 onnxruntime 的输出是否一致")

    input_ids = np.random.randint(
        0,
        20000,
        (1, seq_max_length)
    ).astype(np.int64)

    attn_mask = np.ones(
        (1, seq_max_length),
        dtype=np.int64
    )

    ort_session = ort.InferenceSession(onnx_file)

    ort_outputs = ort_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attn_mask
        }
    )

    ort_prob = ort_outputs[0]
    ort_label = ort_outputs[1]

    input_ids_tensor = torch.tensor(input_ids).to(device)
    attn_mask_tensor = torch.tensor(attn_mask).to(device)

    with torch.no_grad():
        pt_prob, pt_label = model(
            input_ids=input_ids_tensor,
            attention_mask=attn_mask_tensor
        )

        pt_prob = pt_prob.detach().cpu().numpy()
        pt_label = pt_label.detach().cpu().numpy()

    np.testing.assert_allclose(
        pt_prob,
        ort_prob,
        rtol=1e-3,
        atol=1e-5
    )

    np.testing.assert_allclose(
        pt_label,
        ort_label,
        rtol=1e-3,
        atol=1e-5
    )

    logger.info("输出一致，很好")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="pytorch 模型文件夹路径"
    )

    parser.add_argument(
        "--seq_max_length",
        default=512,
        type=int,
        required=True,
        help="最大输入长度"
    )

    parser.add_argument(
        "--num_heads",
        default=12,
        type=int,
        required=True,
        help="保留参数，当前未使用"
    )

    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        required=True,
        help="保留参数，当前未使用"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        required=True,
        help="cpu or cuda"
    )

    parser.add_argument(
        "--onnx_file",
        default=None,
        type=str,
        required=True,
        help="onnx 模型导出路径"
    )

    parser.add_argument(
        "--opt_onnx_file",
        default=None,
        type=str,
        required=False,
        help="优化后的 onnx 模型导出路径，当前未使用"
    )

    parser.add_argument(
        "--pool_type",
        default="first-last-avg",
        type=str,
        choices=["first-last-avg", "cls"],
        help="pooling 方式，必须和训练时保持一致"
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    convert_pt_to_onnx(
        model_dir=args.model_dir,
        device=args.device,
        seq_max_length=args.seq_max_length,
        onnx_file=args.onnx_file,
        pool_type=args.pool_type
    )

    logger.info("job done!")


if __name__ == "__main__":
    main()
