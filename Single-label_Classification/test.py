# 用于单标签的验证集

import os
import re
import torch
import json
import onnx
import time

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from transformers import BertTokenizer
from classifier_model import BertForMultiLabel
# from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForMultiLabel


default_label_file_path = '/user_home/du.jing/国家安全/单标签/data/labels.json'
default_model_file_path = '/user_home/du.jing/国家安全/单标签/model/20260612_165846/best_checkpoint'
default_val_data_path = '/user_home/du.jing/国家安全/单标签/data/val_0616.jsonl'
SCRIPT_DIR = Path(__file__).resolve().parent

def load_model( model_file_path, label_file_path):
    global tokenizer
    global model
    global label2int
    global int2label
    
    # Get Env info
    GPU_ID = 1 
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")
    
    # Load model / label
    model_name = model_file_path.split("/")[-2]
    tokenizer = BertTokenizer.from_pretrained(model_file_path)
    model = BertForMultiLabel.from_pretrained(model_file_path)
    model.to(device)
    model.eval() # 开启评估模式（关闭Dropout等，推理必须加）
    # ort_session = ort.InferenceSession(model_file_path+'/model.onnx')

    label2int = json.load(open(label_file_path,'r',encoding='utf-8'))
    int2label = {v:k for k,v in label2int.items()} 
    print(int2label)


def pre_process_text(text: str) -> str:
    text = re.sub(r"<[a-zA-Z/].*?>", "", text)  # 移除HTML标签（优化匹配方式）
    text = re.sub(r"http[\x00-\x1f\x21-\xff]+", "", text)  # 移除URL
    text = re.sub(r"www\.[\x00-\x1f\x21-\xff]+", "", text)  # 移除www链接
    text = re.sub(r"&nbsp;|&quot;|\u200b|\u3000", "", text)  # 移除特殊空白字符
    text = re.sub(r"\s+", " ", text)  # 多个空格合并为1个
    text = re.sub(r"\n{3,}", "\n", text)  # 3个及以上换行符 → 1个换行符
    text = text.strip()  # 去除首尾空白
    return text

def pre_process_text(text: str) -> str:
    text = re.sub(r"<[a-zA-Z/].*?>", "", text)
    text = re.sub(r"http[\x00-\x1f\x21-\xff]+", "", text)
    text = re.sub(r"www\.[\x00-\x1f\x21-\xff]+", "", text)
    text = re.sub(r"&nbsp;|&quot;|\u200b|\u3000", "", text)
    return text

# 推理
def token2id(text, padding=True):
#     # 文本前后256个token截断。
#     # 此函数不支持batch处理
    inputs = tokenizer(text, max_length=512, padding='max_length')
    if len(inputs['input_ids']) > 512:
        input_ids = [inputs['input_ids'][:256] + inputs['input_ids'][-256:]]
        attention_mask = [inputs['attention_mask'][:512]]
    else:
        if padding:
            input_ids = [inputs['input_ids']]
            attention_mask = [inputs['attention_mask']]
        else:
            input_ids = [[i for i in inputs['input_ids'] if i > 0]]
            attention_mask = [[i for i in inputs['attention_mask'] if i > 0]]

    return input_ids, attention_mask

def predict(input_ids, attention_mask):    

    output = model(
        input_ids=torch.tensor(input_ids).to(model.device), 
        attention_mask=torch.tensor(attention_mask).to(model.device)
    )

    logits = output[0]

    probs = torch.softmax(logits, dim=-1)

    pred_id = torch.argmax(probs, dim=-1).item()
    pred_prob = probs[0][pred_id].item()
    pred_label = int2label[pred_id]

    label_list = pred_label
    prob_list = pred_prob

    return label_list, prob_list

def predict_onnx(input_ids, attention_mask):

    ort_outputs = ort_session.run(
        None, 
        {
            'input_ids': np.array(input_ids), 
            "attention_mask": np.array(attention_mask)
        }
    )
    if len(ort_outputs) >= 2:
        probs = ort_outputs[0]
        labels = ort_outputs[1]

        # prob shape: [batch]
        # label shape: [batch]
        pred_prob = float(probs[0])
        pred_id = int(labels[0])
        pred_label = int2label[pred_id]

    return pred_prob, pred_label

# 预热和测试

def run_warmup():
    text = "资产配置2.0版本（202606） 1、QQQ-20%，721 2、SMH-15%，620 3、CRCL-15%，78 4、科创50-10%，1.756 5、生息类-10%（茅台1290/国债/长江电力28） 6、GOOGL-5%，360 7、SPACX-5%，160 8、期货/合约/期权-8% 9、BTC-3%，63000 9、黄金-3%，4226 10、BNB-3%，609 11、HYPE-3% ，59"
    text = pre_process_text(text)
    input_ids, attention_mask = token2id(text)
    pred_label, pred_prob =  predict(input_ids, attention_mask)
    print(pred_prob)
    print(pred_label)


# Run valset predict
def process_pred(data_path: str, result_path: str):
    data = read_json_or_jsonl(data_path)
    # data = json.load(open(data_path,'r',encoding='utf-8'))

    json_list = []
    for doc in tqdm(data):
        content = doc['content']
        content = pre_process_text(content)

        input_ids, attention_mask = token2id(content)
        pred, prob = predict(input_ids, attention_mask)
        doc['model_pred'] = pred
        doc['model_prob'] = prob
        json_list.append(doc)

    with open(result_path,'w',encoding='utf-8') as f:
        json.dump(json_list,f,indent=4,ensure_ascii=False)
    
    return json_list


# =========================
# 读取 json / jsonl
# =========================

def read_json_or_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    兼容读取：
    1. jsonl：每行一个 JSON 字典
    2. json：标准 JSON 列表文件
    3. json：单个 JSON 字典文件
    4. 即使后缀写错，也会自动尝试整体 JSON / JSONL 两种方式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print(f"[警告] 文件为空: {file_path}")
        return data

    # 优先尝试整体按 JSON 读取
    try:
        raw = json.loads(text)

        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                if not isinstance(item, dict):
                    print(f"[格式错误] 文件: {file_path}, JSON列表索引: {idx}, 不是字典: {type(item)}")
                    continue
                data.append(item)
            return data

        elif isinstance(raw, dict):
            data.append(raw)
            return data

        else:
            raise ValueError(f"JSON根对象既不是列表也不是字典，实际类型: {type(raw)}")

    except Exception:
        pass

    # 如果整体 JSON 失败，再按 JSONL 逐行读取
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[JSONL解析失败] 文件: {file_path}, 行号: {line_idx}, 错误: {repr(e)}")
                continue

            if not isinstance(item, dict):
                print(f"[格式错误] 文件: {file_path}, 行号: {line_idx}, 不是字典: {type(item)}")
                continue

            data.append(item)

    return data


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


# =========================
# 配置区
# =========================

# 真实标签字段
TRUE_LABEL_FIELD = "labels"
# 模型预测字段
PRED_LABEL_FIELD = "model_pred"

def get_label_value(value):
    """
    兼容：
    labels = "类别A"
    labels = ["类别A"]

    单标签任务统一转成字符串。
    """
    if value is None:
        return None

    if isinstance(value, list):
        if len(value) == 0:
            return None
        return str(value[0]).strip()

    return str(value).strip()


def count_label_distribution(data, label_field):
    """
    统计某个字段下每个类别的数量。
    """
    counter = Counter()
    missing_count = 0

    for idx, item in enumerate(data):
        if label_field not in item:
            missing_count += 1
            continue

        label = get_label_value(item.get(label_field))

        if label is None or label.strip() == "":
            missing_count += 1
            continue

        counter[label] += 1

    return counter, missing_count


def extract_true_and_pred(valid_data):
    """
    从验证集预测结果中提取真实标签和预测标签。
    """
    y_true = []
    y_pred = []
    skip_count = 0

    for idx, item in enumerate(valid_data):
        if TRUE_LABEL_FIELD not in item:
            print(f"[缺少字段] 验证集第 {idx} 条缺少 {TRUE_LABEL_FIELD}")
            skip_count += 1
            continue

        if PRED_LABEL_FIELD not in item:
            print(f"[缺少字段] 验证集第 {idx} 条缺少 {PRED_LABEL_FIELD}")
            skip_count += 1
            continue

        true_label = get_label_value(item.get(TRUE_LABEL_FIELD))
        pred_label = get_label_value(item.get(PRED_LABEL_FIELD))

        if true_label is None or true_label == "" or pred_label is None or pred_label == "":
            print(f"[空标签] 验证集第 {idx} 条: labels={true_label}, model_pred={pred_label}")
            skip_count += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

    print(f"验证集有效评估样本数: {len(y_true)}")
    print(f"验证集跳过样本数: {skip_count}")

    return y_true, y_pred


def save_overall_metrics_txt(
    output_path,
    accuracy,
    macro_p,
    macro_r,
    macro_f1,
    micro_p,
    micro_r,
    micro_f1,
    weighted_p,
    weighted_r,
    weighted_f1
):
    """
    保存整体指标到 txt。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("========== 整体指标 ==========\n")
        f.write(f"accuracy    : {accuracy:.6f}\n")
        f.write(f"macro_p     : {macro_p:.6f}\n")
        f.write(f"macro_r     : {macro_r:.6f}\n")
        f.write(f"macro_f1    : {macro_f1:.6f}\n")
        f.write(f"micro_p     : {micro_p:.6f}\n")
        f.write(f"micro_r     : {micro_r:.6f}\n")
        f.write(f"micro_f1    : {micro_f1:.6f}\n")
        f.write(f"weighted_p  : {weighted_p:.6f}\n")
        f.write(f"weighted_r  : {weighted_r:.6f}\n")
        f.write(f"weighted_f1 : {weighted_f1:.6f}\n")

def save_overall_metrics_csv(
    output_path,
    accuracy,
    macro_p,
    macro_r,
    macro_f1,
    micro_p,
    micro_r,
    micro_f1,
    weighted_p,
    weighted_r,
    weighted_f1
):
    """
    保存整体指标到 csv。
    """
    rows = [
        {"metric": "accuracy", "value": round(float(accuracy), 6)},
        {"metric": "macro_p", "value": round(float(macro_p), 6)},
        {"metric": "macro_r", "value": round(float(macro_r), 6)},
        {"metric": "macro_f1", "value": round(float(macro_f1), 6)},
        {"metric": "micro_p", "value": round(float(micro_p), 6)},
        {"metric": "micro_r", "value": round(float(micro_r), 6)},
        {"metric": "micro_f1", "value": round(float(micro_f1), 6)},
        {"metric": "weighted_p", "value": round(float(weighted_p), 6)},
        {"metric": "weighted_r", "value": round(float(weighted_r), 6)},
        {"metric": "weighted_f1", "value": round(float(weighted_f1), 6)},
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def set_chinese_font():
    """
    根据操作系统设置合适的中文字体
    """
    
    import matplotlib.pyplot as plt
    import platform
    
    system_name = platform.system()
    if system_name == "Windows":
        # Windows 常用黑体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
    elif system_name == "Darwin":  # Mac OS
        # Mac 常用字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
    else:  # Linux
        # Linux 通常需要安装字体，如 WenQuanYi
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    # 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

def save_confusion_matrix(cm, classes, output_path, normalize=False, title='Confusion Matrix'):
    """
    绘制并保存混淆矩阵
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    set_chinese_font()
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm_normalized
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        cm_to_plot = cm
        fmt = 'd'
    
    plt.figure(figsize=(max(10, len(classes)*0.8), max(8, len(classes)*0.6)))
    
    sns.heatmap(
        cm_to_plot, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵图已保存")


def get_args():
    parser = ArgumentParser(description="Process valset evaluation")
    parser.add_argument("--model-path", type=str, default=default_model_file_path, help="Model to run evaluation")
    parser.add_argument("--eval-data", type=str, default=default_val_data_path, help="Valuation dataset path")
    parser.add_argument("--result-suffix", type=str, default="", help="Valuation dataset path")
    parser.add_argument("--label-path", type=str, default=default_label_file_path, help="Label to index mapping file path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR),
        help="测试输出根目录；默认是 test.py 所在目录",
    )
    parser.add_argument("--save-result-csv", type=str_to_bool, default=False, help="")
    parser.add_argument("--plot-confusion-matrix", type=str_to_bool, default=False, help="")
    
    args = parser.parse_args()
    return args

    

def main():
    opts = get_args()
    eval_data_path = str(Path(opts.eval_data).expanduser().resolve())
    model_path = str(Path(opts.model_path).expanduser().resolve())
    label_path = str(Path(opts.label_path).expanduser().resolve())
    output_root = Path(opts.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    save_result_csv = opts.save_result_csv
    plot_confusion_matrix = opts.plot_confusion_matrix
    
    load_model(model_file_path=model_path, label_file_path=label_path)
    run_warmup()
    
    # 输出目录
    if not opts.result_suffix:
        predict_save_path = f"{Path(eval_data_path).stem}_preded_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    else:
        predict_save_path = f"{Path(eval_data_path).stem}_preded_{opts.result_suffix}.json"
    output_dir = output_root / f"eval_result_{Path(predict_save_path).stem}"
    predict_save_path = output_dir / predict_save_path

    # 输出文件名
    PER_CLASS_XLSX = "per_class_metrics.xlsx"
    PER_CLASS_CSV = "per_class_metrics.csv"
    OVERALL_TXT = "overall_metrics.txt"
    OVERALL_CSV = "overall_metrics.csv"
    CONFUSION_MAT = "confusion_matrix.png"
    CONFUSION_CSV = "confusion_matrix.csv"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== 获取验证集预测结果 ==========")
    if predict_save_path.exists():
        valid_data = read_json_or_jsonl(predict_save_path)
        print(f"已从缓存读取验证样本数: {len(valid_data)}")
    else:
        valid_data = process_pred(eval_data_path, predict_save_path)
        print(f"从验证集执行预测并缓存: {len(valid_data)}")

    print("\n========== 统计验证集真实类别数量 ==========")
    valid_true_count, valid_true_missing = count_label_distribution(
        valid_data,
        TRUE_LABEL_FIELD
    )
    print(f"验证集缺少真实标签数量: {valid_true_missing}")

    print("\n========== 统计验证集预测类别数量 ==========")
    valid_pred_count, valid_pred_missing = count_label_distribution(
        valid_data,
        PRED_LABEL_FIELD
    )
    print(f"验证集缺少预测标签数量: {valid_pred_missing}")

    print("\n========== 提取真实标签和预测标签 ==========")
    y_true, y_pred = extract_true_and_pred(valid_data)

    if len(y_true) == 0:
        print("[错误] 没有有效验证样本，无法计算指标。")
        return

    # 只统计验证集真实标签中出现过的类别
    # 也就是“验证集里有的标签”
    eval_labels = sorted(set(y_true))

    print(f"\n验证集中真实出现的类别数: {len(eval_labels)}")

    print("\n========== 计算整体指标 ==========")

    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        average="macro",
        zero_division=0
    )

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        average="micro",
        zero_division=0
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        average="weighted",
        zero_division=0
    )

    print(f"accuracy    : {accuracy:.6f}")
    print(f"macro_p     : {macro_p:.6f}")
    print(f"macro_r     : {macro_r:.6f}")
    print(f"macro_f1    : {macro_f1:.6f}")
    print(f"micro_p     : {micro_p:.6f}")
    print(f"micro_r     : {micro_r:.6f}")
    print(f"micro_f1    : {micro_f1:.6f}")
    print(f"weighted_p  : {weighted_p:.6f}")
    print(f"weighted_r  : {weighted_r:.6f}")
    print(f"weighted_f1 : {weighted_f1:.6f}")

    print("\n========== 计算每个类别指标 ==========")

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        average=None,
        zero_division=0
    )

    # 每个类别预测正确数量
    correct_count = Counter()
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_count[true_label] += 1

    rows = []

    for label, p, r, f1, support in zip(eval_labels, per_p, per_r, per_f1, per_support):
        valid_true_num = valid_true_count.get(label, 0)
        valid_pred_num = valid_pred_count.get(label, 0)
        valid_correct_num = correct_count.get(label, 0)

        row = {
            "类别": label,

            # 数量统计
            "验证集真实数量": valid_true_num,
            "验证集预测数量": valid_pred_num,
            "验证集预测正确数量": valid_correct_num,

            # 指标
            "Precision": round(float(p), 6),
            "Recall": round(float(r), 6),
            "F1": round(float(f1), 6),
            "Support": int(support),
        }

        rows.append(row)
    
    df = pd.DataFrame(rows)

    # 按验证集真实数量从多到少排序
    df = df.sort_values(by="验证集真实数量", ascending=False)
    
    # =========== 打印和保存测试结果 ===========

    print("\n========== 每个类别统计结果 ==========")
    print(df.to_string(index=False))

    # 保存每类指标 Excel
    xlsx_path = output_dir / PER_CLASS_XLSX
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    
    # 保存每类指标 CSV
    if save_result_csv:
        csv_path = output_dir / PER_CLASS_CSV
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存整体指标 TXT
    overall_txt_path = output_dir / OVERALL_TXT
    save_overall_metrics_txt(
        output_path=overall_txt_path,
        accuracy=accuracy,
        macro_p=macro_p,
        macro_r=macro_r,
        macro_f1=macro_f1,
        micro_p=micro_p,
        micro_r=micro_r,
        micro_f1=micro_f1,
        weighted_p=weighted_p,
        weighted_r=weighted_r,
        weighted_f1=weighted_f1
    )

    # 保存整体指标 CSV
    if save_result_csv:
        overall_csv_path = output_dir / OVERALL_CSV
        save_overall_metrics_csv(
            output_path=overall_csv_path,
            accuracy=accuracy,
            macro_p=macro_p,
            macro_r=macro_r,
            macro_f1=macro_f1,
            micro_p=micro_p,
            micro_r=micro_r,
            micro_f1=micro_f1,
            weighted_p=weighted_p,
            weighted_r=weighted_r,
            weighted_f1=weighted_f1
        )

    # 保存混淆矩阵
    # 混淆矩阵也只用验证集中真实出现过的标签
    cm = confusion_matrix(y_true, y_pred, labels=eval_labels)
    cm_df = pd.DataFrame(cm, index=eval_labels, columns=eval_labels)
    confusion_image_path = output_dir / CONFUSION_MAT
    if plot_confusion_matrix:
        save_confusion_matrix(cm, eval_labels, confusion_image_path)

    confusion_path = output_dir / CONFUSION_CSV
    cm_df.to_csv(confusion_path, encoding="utf-8-sig")

    print("\n========== 保存完成 ==========")
    print(f"预测缓存 JSON   : {predict_save_path}")
    print(f"每个类别指标 Excel: {xlsx_path}")
    if save_result_csv:
        print(f"每个类别指标 CSV : {csv_path}")
    print(f"整体指标 TXT     : {overall_txt_path}")
    if save_result_csv:
        print(f"整体指标 CSV     : {overall_csv_path}")
    print(f"混淆矩阵 CSV     : {confusion_path}")
    if plot_confusion_matrix:
        print(f"混淆矩阵图片    : {confusion_image_path}")


if __name__ == "__main__":
    main()
