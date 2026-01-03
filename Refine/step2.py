## 第二步：调用llama3-70B 对预测输出进行评分和评价

import openai
import json
import concurrent.futures
import threading
import os
import time
import random
import argparse  # 导入 argparse
from tqdm import tqdm # 导入 tqdm

API_KEYS = [
    "fake-key1", "fake-key2", "fake-key3", 
    "fake-key4", "fake-key5", "fake-key6",
    "fake-key7", "fake-key8", "fake-key9", 
    "fake-key10", "fake-key11", "fake-key12",
    "fake-key13", "fake-key14", "fake-key15", 
    "fake-key16", "fake-key17", "fake-key18",
    "fake-key1", "fake-key2", "fake-key3", 
    "fake-key4", "fake-key5", "fake-key6",
    "fake-key7", "fake-key8", "fake-key9", 
    "fake-key10", "fake-key11", "fake-key12",
    "fake-key13", "fake-key14", "fake-key15", 
    "fake-key16", "fake-key17", "fake-key18",
    "fake-key1", "fake-key2", "fake-key3", 
    "fake-key4", "fake-key5", "fake-key6",
    "fake-key7", "fake-key8", "fake-key9", 
    "fake-key10", "fake-key11", "fake-key12",
    "fake-key13", "fake-key14", "fake-key15", 
    "fake-key16", "fake-key17", "fake-key18"
]

# ------------------------------------------------
# 移除了全局的 API_BASE, MODEL_NAME, TEMPERATURE, MAX_TOKENS, TIMEOUT, MAX_RETRIES
# 它们将作为参数传入
# ------------------------------------------------

write_lock = threading.Lock()
counter_lock = threading.Lock()
counter = 0  # 全局计数器

def chat_with_llama(
    api_key, 
    user_prompt,
    api_base, 
    model_name, 
    temperature, 
    max_tokens, 
    timeout, 
    max_retries
):
    """
    调用与 OpenAI ChatCompletion 兼容的本地 Llama 服务。
    使用传入的 API 配置。
    """
    openai.api_base = api_base
    openai.api_key = api_key

    # (System prompt 保持硬编码，因为它是此脚本的核心逻辑)
    system_prompt = '''You are a strict and objective evaluator. Given an Instruction, Input, Reference Answer, and a Candidate Answer, your task is to:
1. Assign a score (integer, 1–10).
2. Provide a concise but detailed evaluation with clear, actionable feedback.
... (省略 system prompt 的剩余部分) ...
- Suggestions: <3–5 specific, actionable improvements>
'''

    for attempt in range(1, max_retries + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=timeout,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = f"ERROR: {str(e)}"
            if attempt < max_retries:
                time.sleep(0.5 * attempt + random.random() * 0.5)
            else:
                return err

def process_item(
    item, 
    api_key, 
    output_path, 
    round_tag_field_instruction, 
    round_tag_field_input, 
    round_tag_field_output_model, 
    round_tag_field_evaluation,
    api_config # 传入 API 配置字典
):
    """
    (函数逻辑保持不变)
    """
    global counter

    instruction = item.get(round_tag_field_instruction, "")
    input_str = item.get(round_tag_field_input, "") # 
    output = item.get("output_round_0", "")
    output_model = item.get(round_tag_field_output_model, "")
    
    user_prompt = f'''Now given:\nInstruction: {instruction}\nInput: {input_str}\nReference Answer: {output}\nCandidate Answer: {output_model}\n\nScore:\nEvaluation:'''

    # 
    response = chat_with_llama(
        api_key, 
        user_prompt,
        **api_config # 
    )

    item[round_tag_field_evaluation] = response

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with counter_lock:
        counter += 1
        current_id = counter
    # 
    # 
    # 

def main(
    input_path, 
    output_path, 
    round_tag_field_instruction, 
    round_tag_field_input, 
    round_tag_field_output_model, 
    round_tag_field_evaluation,
    api_config 
):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    open(output_path, "w", encoding="utf-8").close()

    total_items = len(data)
    if total_items == 0:
        print("No items to process.")
        return
        
    print(f"Starting processing of {total_items} items...")
    loop_start_time = time.time()
    items_processed = 0
    update_interval = max(1, total_items // 100) # 


    with concurrent.futures.ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
        futures = []
        for idx, item in enumerate(data):
            api_key = API_KEYS[idx % len(API_KEYS)]
            futures.append(
                executor.submit(
                    process_item, 
                    item, 
                    api_key, 
                    output_path, 
                    round_tag_field_instruction, 
                    round_tag_field_input, 
                    round_tag_field_output_model, 
                    round_tag_field_evaluation,
                    api_config # 
                )
            )
        
        with tqdm(concurrent.futures.as_completed(futures), total=total_items, desc="Processing items") as pbar:
            for future in pbar:
                items_processed += 1
                
                try:
                    future.result()  # 
                except Exception as e:
                    tqdm.write(f"A task generated an exception: {e}")

                if items_processed == 1 or items_processed % update_interval == 0 or items_processed == total_items:
                    time_elapsed = time.time() - loop_start_time
                    avg_time_per_item = time_elapsed / items_processed
                    items_remaining = total_items - items_processed
                    etr_seconds = items_remaining * avg_time_per_item
                    
                    etr_h = int(etr_seconds // 3600)
                    etr_m = int((etr_seconds % 3600) // 60)
                    etr_s = int(etr_seconds % 60)
                    
                    pbar.set_postfix_str(
                        f"Avg: {avg_time_per_item:.2f}s/item, ETR: {etr_h:02d}h {etr_m:02d}m {etr_s:02d}s"
                    )

def jsonl_to_json(jsonl_path, json_path):
    data = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        print(f"Error: Could not find {jsonl_path} to convert.")
    except Exception as e:
        print(f"Error during JSONL to JSON conversion: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate model outputs using a local LLM API.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for input and output.")
    parser.add_argument("--Name", type=str, required=True, help="Name (e.g., timestamp) for this run, used in file paths.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens to generate.")
    parser.add_argument("--timeout", type=int, default=15, help="API request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries on API failure.")
    
    args = parser.parse_args()

    API_BASE = os.getenv("Strong_API_BASE", "")
    MODEL_NAME = os.getenv("Strong_MODEL_NAME", "")
    api_config = {
        "api_base": API_BASE,
        "model_name": MODEL_NAME,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_retries": args.max_retries
    }

    round_tag = "round_0"
    round_tag_field_instruction = f"instruction_{round_tag}"
    round_tag_field_input = f"input_{round_tag}"
    round_tag_field_output_model = f"output_model_{round_tag}"
    round_tag_field_evaluation = f"evaluation_{round_tag}"
    

    input_path = os.path.join(args.base_dir, args.Name, "step1_output", "Low.json")
    output_path = os.path.join(args.base_dir, args.Name, "step2_output", "Low.jsonl")
    final_output_path = os.path.join(args.base_dir, args.Name, "step2_output", "Low.json")

    print(f"Starting evaluation...")
    print(f"Input file: {input_path}")
    print(f"Temporary output (JSONL): {output_path}")
    print(f"Final output (JSON): {final_output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if final_output_path:
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    total_start_time = time.time()

    main(
        input_path, 
        output_path, 
        round_tag_field_instruction, 
        round_tag_field_input, 
        round_tag_field_output_model, 
        round_tag_field_evaluation,
        api_config 
    )

    print(f"\nMain processing complete.")
    print("Converting JSONL to JSON...")
    
    jsonl_to_json(output_path, final_output_path)
    
    print(f"Conversion complete. Final output file: {final_output_path}")

    # 
    total_end_time = time.time()
    total_elapsed_seconds = total_end_time - total_start_time
    total_m = int((total_elapsed_seconds % 3600) // 60)
    total_s = int(total_elapsed_seconds % 60)
    total_h = int(total_elapsed_seconds // 3600)
    print(f"Total execution time: {total_h}h {total_m}m {total_s}s")
