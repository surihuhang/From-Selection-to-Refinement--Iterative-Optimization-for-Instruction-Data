## Step 3: Bottom-up reflection (root cause analysis) to update instructions and inputs

import openai
import json
import concurrent.futures
import threading
import os
import time
import random
import argparse  # Import argparse
from tqdm import tqdm # Import tqdm

# ------------------ Configuration Area ------------------
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
    "fake-key16", "fake-key17", "fake-key18"
]

# (Global API_BASE, MODEL_NAME, etc., are removed. They will be passed as arguments.)
# ------------------------------------------------

write_lock = threading.Lock()
counter_lock = threading.Lock()
counter = 0  # Global counter

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
    Calls a local Llama service compatible with OpenAI ChatCompletion.
    Includes simple retries, returns plain text answer or 'ERROR: ...'
    """
    openai.api_base = api_base
    openai.api_key = api_key

    system_prompt = '''You will be given the following information:
• The original instruction
• The original input (may be empty)
• The original reference output
• Model’s output
• Evaluation of the model’s output (including score 1–10 and feedback)

Your task is:
1. Analyze the model’s output and its evaluation to determine whether issues stem from unclear, incomplete, or poorly targeted instructions/inputs.
2. Identify possible improvement directions for the original instructions/inputs, such as clearer formatting rules, richer context, closer task alignment, reduced ambiguity, or adding a few ICL examples.
3. Revise the instruction and input based on this analysis so that future models achieve higher scores and more positive feedback.
4. Keep the output format strict and minimal.

Output format(strict, no deviations):
instruction_new: <revised instruction, or keep original if no changes needed>
input_new:<if original input is empty, this line must end immediately after the colon with nothing following it>

Important Rules:
- Preserve all essential content, such as links, code or tables, from the original instruction or input; do not remove or simplify them.
- The revised instruction and input must not contain any meta words such as "reference answer", "reference output", "revise", "Notes", "Improvements", or anything similar. They should read as natural standalone task data.
- Only output the two fields shown above. No explanations, comments, or additional sections.
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
            # Back off and retry
            if attempt < max_retries:
                time.sleep(0.5 * attempt + random.random() * 0.5)
            else:
                return err

# (These imports are assumed to be in the same directory)
from step3_process import step3_process_transform_json
from step4 import step4_process_item
from step5 import step5_process_item
from step5_process import step5_process_transform_json
from step6 import step6_process_item
from step6_process import step6_transform_json


def process_item(item, api_key, output_path, Result_output_path, api_config):
    """
    Processes a single sample:
    - Assembles instruction + input
    - Calls the model to get a response
    - Writes the response to item['output_model']
    - Writes the item to the target JSONL file row by row
    """
    global counter

    instruction = item.get("instruction_round_0", "")
    input_str = item.get("input_round_0", "") # Renamed to avoid conflict with 'input' function
    output = item.get("output_round_0", "")
    output_model = item.get("output_model_round_0", "")
    evaluation = item.get("evaluation_round_0", "")

    Winner = True # Initialize loop condition
    # Used to store the last successful result from the loop
    last_successful_instruction = None
    last_successful_input = None

    round_num = 1
    while Winner and (round_num < 4):
        if round_num - 1 == 0:
            user_prompt = f'''Now given:\nThe original instruction: {instruction}\nThe original input: {input_str}\nThe original reference output: {output}\nModel's output: {output_model}\nEvaluation of the model’s output: {evaluation}\n\ninstruction_new:\ninput_new:'''
        else:
            # [Requirement Update] Dynamically get the evaluation feedback from the previous round...
            for i in range(round_num - 1, -1, -1):
                prev_round_evaluation_key = f"evaluation_round_{i}"
                if round_num - 1 == 0:
                    user_prompt = f'''Now given:\nThe original instruction: {instruction}\nThe original input: {input_str}\nThe original reference output: {output}\nModel's output: {output_model}\nEvaluation of the model’s output: {evaluation}\n\ninstruction_new:\ninput_new:'''
                    break
                if item.get(prev_round_evaluation_key):
                    Judge_feedback = item.get(prev_round_evaluation_key)
                    user_prompt = f'''Now given:\nThe original instruction: {instruction}\nThe original input: {input_str}\nThe original reference output: {output}\nModel's output: {output_model}\nEvaluation of the model’s output: {evaluation}\nAdditional information: The previous revision was judged to be worse than the original. Expert feedback on the quality difference between the original data (A) and the revised data (B) is provided for your reference: {Judge_feedback}\n\ninstruction_new:\ninput_new:'''
                    break # Break the loop after finding the most recent feedback
        
        # (Step 3) First generate modify_instruction
        response = chat_with_llama(api_key, user_prompt, **api_config)

        # (Step 3_process) Process the response
        new_instruction, new_input = step3_process_transform_json(response)
        
        # Note!!! If no match, continue the loop
        if new_instruction is None or new_input is None:
            tqdm.write(f"Warning: Failed to process response in round {round_num} for an item. Skipping to next iteration.")
            round_num += 1 # Must increment round_num, otherwise it might cause an infinite loop
            continue
        
        # (Step 4) baseLLM generates model_output_num
        new_model_output = step4_process_item(new_instruction, new_input)

        # (Step 5) sLLM judges win/lose/tie
        judge_result = step5_process_item(instruction, input_str, output_model, new_instruction, new_input, new_model_output)
        # (Step 5_process) Process judge_result
        Winner, new_evaluation = step5_process_transform_json(judge_result)

        # Note!!! If no match, continue the loop
        if Winner is None:
            tqdm.write(f"Warning: Failed to process evaluation in round {round_num} for an item. Skipping to next iteration.")
            round_num += 1 # Must increment round_num, otherwise it might cause an infinite loop
            continue
            
        item[f'instruction_round_{round_num}'] = new_instruction
        item[f'input_round_{round_num}'] = new_input
        item[f'output_model_round_{round_num}'] = new_model_output
        item[f'evaluation_round_{round_num}'] = new_evaluation
        
        # Store the last successful iteration's result
        last_successful_instruction = new_instruction
        last_successful_input = new_input
        round_num += 1

    # Ensure the loop was successful at least once
    if last_successful_instruction is not None:
        output_new_response = step6_process_item(last_successful_instruction, last_successful_input, output)
        # Process output_new, assuming a result is always returned
        output_new = step6_transform_json(output_new_response)
        item['output_new'] = output_new
    else:
        # If loop never succeeded, fall back to original
        last_successful_instruction = instruction
        last_successful_input = input_str
        output_new = output
        item['output_new'] = output_new # Ensure output_new exists
    
    # Write file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result = {
            "instruction": last_successful_instruction,
            "input": last_successful_input,
            "output": output_new
        }

    with write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Only take the finally revised {instruct, input, output} and append to the JSONL file
        with open(Result_output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Update global counter
    with counter_lock:
        counter += 1
        current_id = counter

def main(input_path, output_path, Result_output_path, api_config):
    """
    Main processing function with time estimation.
    """
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
    
    # Ensure Result_output_path directory exists and file is cleared
    result_output_dir = os.path.dirname(Result_output_path)
    if result_output_dir and not os.path.exists(result_output_dir):
        os.makedirs(result_output_dir, exist_ok=True)
    open(Result_output_path, "w", encoding="utf-8").close()


    # --- Add time estimation setup ---
    total_items = len(data)
    if total_items == 0:
        print("No items to process.")
        return
        
    print(f"Starting processing of {total_items} items...")
    loop_start_time = time.time()
    items_processed = 0
    # Update ETR roughly every 1%
    update_interval = max(1, total_items // 100) 
    # --- End setup ---

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
        futures = []
        for idx, item in enumerate(data):
            api_key = API_KEYS[idx % len(API_KEYS)]
            futures.append(
                executor.submit(process_item, item, api_key, output_path, Result_output_path, api_config)
            )
        
        # --- Replace wait() with tqdm(as_completed()) for time estimation ---
        with tqdm(concurrent.futures.as_completed(futures), total=total_items, desc="Processing items") as pbar:
            for future in pbar:
                items_processed += 1
                
                try:
                    future.result()  # Check for exceptions
                except Exception as e:
                    tqdm.write(f"A task generated an exception: {e}")
                
                # Update ETR
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
        # --- End replacement ---

def jsonl_to_json(jsonl_path, json_path):
    """Converts a JSONL file to a standard JSON array file"""
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
    
    # --- Argparse Setup ---
    parser = argparse.ArgumentParser(description="Step 3: Reflect and revise instructions using a local LLM API.")
    
    # Path and Name arguments
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for input and output.")
    parser.add_argument("--Name", type=str, required=True, help="Name (e.g., timestamp) for this run, used in file paths.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate.")
    parser.add_argument("--timeout", type=int, default=15, help="API request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries on API failure.")
    parser.add_argument("--Result_Path", type=str, required=True, help="Base directory for output.")
    
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

    # Use arguments to build paths
    input_path = os.path.join(args.base_dir, args.Name, "step2_output", "Low.json")
    output_path = os.path.join(args.base_dir, args.Name, "step3_output", "Low.jsonl")
    final_output_path = os.path.join(args.base_dir, args.Name, "step3_output", "Low.json")

    Result_output_path = os.path.join(args.base_dir, args.Name, "step3_output", "output.jsonl")
    Result_output_path_json = os.path.join(args.base_dir, args.Name, "step3_output", "output.json")
    Result_Path = args.Result_Path
    print(f"Starting revision (Step 3)...")
    print(f"Input file: {input_path}")
    print(f"Full log output (JSONL): {output_path}")
    print(f"Final revised data (JSONL): {Result_output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if final_output_path:
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    if Result_output_path_json:
        os.makedirs(os.path.dirname(Result_output_path_json), exist_ok=True)
    if Result_Path:
        os.makedirs(os.path.dirname(Result_Path), exist_ok=True)

    # --- Add total time calculation ---
    total_start_time = time.time()

    main(input_path, output_path, Result_output_path, api_config)

    print(f"\nMain processing complete.")
    print("Converting full log (JSONL) to JSON...")
    jsonl_to_json(output_path, final_output_path)
    print(f"Conversion complete. Output file: {final_output_path}")

    print("Converting final revised data (JSONL) to JSON...")
    jsonl_to_json(Result_output_path, Result_output_path_json)
    print(f"Conversion complete. Output file: {Result_output_path_json}")

    jsonl_to_json(Result_output_path, Result_Path)
    print(f"Conversion complete. Output file: {Result_Path}")

    # Print total time
    total_end_time = time.time()
    total_elapsed_seconds = total_end_time - total_start_time
    total_m = int((total_elapsed_seconds % 3600) // 60)
    total_s = int(total_elapsed_seconds % 60)
    total_h = int(total_elapsed_seconds // 3600)
    print(f"Total execution time: {total_h}h {total_m}m {total_s}s")
    # --- End total time calculation ---
