import json
import re
import os

# def extract_field(text, field):
#     """
#     从类似 'instruction_new: xxx\ninput_new: yyy' 这样的字符串中提取字段
#     """
#     pattern = rf"{field}:\s*(.*)"
#     match = re.search(pattern, text)
#     if match:
#         return match.group(1).strip()
#     return None


def extract_field(text, field):
    if field == "instruction_new":
        # 到下一段 input_new: 或 文本结尾 为止（都支持）
        pattern = r"instruction_new:\s*([\s\S]*?)(?=\r?\ninput_new:|$)"
    elif field == "input_new":
        # input_new: 后面到文本结尾
        pattern = r"input_new:\s*([\s\S]*)"
    elif field == "output_new":
        # output_new: 后面到文本结尾
        pattern = r"output_new:\s*([\s\S]*)"
    else:
        raise ValueError("只支持 instruction_new, input_new和output_new 三个字段")
    
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None



def step3_process_transform_json(modified_text):

    # 抽取 instruction 和 input
    instruction_new = extract_field(modified_text, "instruction_new")
    input_new = extract_field(modified_text, "input_new")

    return instruction_new, input_new
