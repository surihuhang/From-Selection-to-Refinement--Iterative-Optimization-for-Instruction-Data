## 第四步：调用llama3-70B 进行自上而下的进化，根据更新的指令和输入，改写参考答案

import openai
import json
import concurrent.futures
import threading
import os
import time
import random

# ------------------ 可配置区 ------------------
# API_KEYS = [
#     "fake-key1", "fake-key2", "fake-key3", 
#     "fake-key4", "fake-key5", "fake-key6",
#     "fake-key7", "fake-key8", "fake-key9", 
#     "fake-key10", "fake-key11", "fake-key12",
#     "fake-key13", "fake-key14", "fake-key15", 
#     "fake-key16", "fake-key17", "fake-key18",
#     "fake-key19", "fake-key20", "fake-key21", 
#     "fake-key22", "fake-key23", "fake-key24",
#     "fake-key25", "fake-key26", "fake-key27", 
#     "fake-key28", "fake-key29", "fake-key30",
#     "fake-key31", "fake-key32", "fake-key33", 
#     "fake-key34", "fake-key35", "fake-key36",
# ]

api_key = "fake-key1"

API_BASE = os.getenv("Strong_API_BASE", "")
MODEL_NAME = os.getenv("Strong_MODEL_NAME", "")
TEMPERATURE = 0.6
MAX_TOKENS = 4096
TIMEOUT = 15
MAX_RETRIES = 3
# ------------------------------------------------

write_lock = threading.Lock()
counter_lock = threading.Lock()
counter = 0  # 全局计数器

def chat_with_llama(api_key, user_prompt):
    """
    调用与 OpenAI ChatCompletion 兼容的本地 Llama3-8B 服务。
    带简单重试，返回纯文本答案或 'ERROR: ...'
    """
    openai.api_base = API_BASE
    openai.api_key = api_key

    system_prompt = '''You will be given the following information:
•	The modified instruction (instruction_new)
•	The modified input (input_new, may be empty)
•	The original reference output

Your task is:
1. Carefully read the modified instruction and input.
2. Using the modified instruction and input, revise the original reference output, ensuring:
•	Correctness: answer aligns with Instruction/Input/Reference; no hallucinations.
•	Completeness: covers all required points; no major omissions.
•	Clarity: uses clear, concise, unambiguous language.
•	Instruction & Format Compliance: follows required structure, style, and rules.
•	Reasoning Quality: Logical, coherent, and justified reasoning where needed.

Output format(strict, no deviations):
output_new: <revised output>  

Important Rules:
- Revise the original reference output without altering its correctness. Introducing errors can cause negative impact.
- The revised output must not contain any meta words such as "reference output", "revise", "Notes", "Improvements", or anything similar. They should read as natural standalone task data.
- Only output the field shown above. No explanations, comments, or additional sections.
'''

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n=1,
                timeout=TIMEOUT,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = f"ERROR: {str(e)}"
            # 退避一会儿再试
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * attempt + random.random() * 0.5)
            else:
                return err

def step6_process_item(last_successful_instruction, last_successful_input, output):
    """
    对单条样本：
    - 拼接 instruction + input
    - 调用模型拿回答
    - 将回答写入 item['output_model']
    - 逐行写入目标 JSONL 文件
    """
    global counter

    instruction = last_successful_instruction
    input = last_successful_input
    output = output
    
    user_prompt = f'''Now given:\nThe modified instruction and input: \nThe modified instruction: {instruction}\nThe modified input: {input}\nThe original reference output: {output}\n\noutput_new:'''

    response = chat_with_llama(api_key, user_prompt)

    # print(f"Processed ID {current_id}\n")
    # print(response)
    return response