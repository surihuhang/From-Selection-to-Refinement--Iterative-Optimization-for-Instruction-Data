import openai
import json
import concurrent.futures
import threading
import os
import time
import argparse  
import re        
from tqdm import tqdm


API_KEYS = [
    "fake-key1", "fake-key2", "fake-key3", 
    "fake-key4", "fake-key5", "fake-key6",
    "fake-key34", "fake-key35", "fake-key36",
    "fake-key1", "fake-key2", "fake-key3", 
    "fake-key4", "fake-key5", "fake-key6",
    "fake-key34", "fake-key35", "fake-key36",
    "fake-key1", "fake-key2", "fake-key3", 
    "fake-key4", "fake-key5", "fake-key6",
    "fake-key34", "fake-key35", "fake-key36",
]


low_file_lock = threading.Lock()
high_file_lock = threading.Lock()
write_lock = threading.Lock() 

def chat_with_gpt(api_key, api_base, model_name, system_prompt, user_prompt):
    openai.api_base = api_base
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2000,
        temperature=0.6,
        n=1,
        timeout=15,
    )
    return response['choices'][0]['message']['content'].strip()


def process_item(item, api_key, api_base, model_name, output_dir, system_prompt, user_prompt_template):
    try:
        question = item["instruction"] + item["input"]
        answer = item["output"]
        user_prompt = user_prompt_template.format(QUESTION=question, ANSWER=answer)

        # 1. Call API
        response = chat_with_gpt(api_key, api_base, model_name, system_prompt, user_prompt)
        
        # --- Core change: Parsing and splitting logic ---
        score_text = response
        # 2. Check for "No"
        if 'RESPONSE: No' in score_text:
            return f"ID {item.get('sample_id', 'N/A')} skipped (RESPONSE: No)."

        # 3. Extract score using regex
        match = re.search(r'Score:\s*(\d+)', score_text)
        
        output_file_path = None
        lock_to_use = None
        score_val = "N/A"

        if match:
            
            score = match.group(1)
            item["score"] = int(score)
            score_val = score
            output_file_path_all = os.path.join(output_dir, 'Data_stage_1_all.json')
            if score == '1':
                # Score is 1, save to low file
                output_file_path = os.path.join(output_dir, 'Data_stage_1_low.json')
                lock_to_use = low_file_lock
            elif score == '2':
                # Score is 2, save to high file
                output_file_path = os.path.join(output_dir, 'Data_stage_1_high.json')
                lock_to_use = high_file_lock
            else:
                # Matched a score, but it's not 1 or 2
                return f"ID {item.get('sample_id', 'N/A')} skipped (Score not 1 or 2: {score})."
        else:
            # "Score: X" not found
            return f"ID {item.get('sample_id', 'N/A')} skipped (Score pattern not found in response)."

        # 4. 
        # (Directory already created in main function)

        # 5. 
        with lock_to_use:
            with open(output_file_path, 'a', encoding='utf-8') as f:
                # Write in jsonl format (one json per line)
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with write_lock:
            with open(output_file_path_all, 'a', encoding='utf-8') as f:
                # Write in jsonl format (one json per line)
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return f"ID {item.get('sample_id', 'N/A')} processed. Score: {score_val}. Saved to {os.path.basename(output_file_path)}"
    
    except Exception as e:
        return f"Error processing ID {item.get('sample_id', 'N/A')}: {e}"


def main(input_path, output_dir, system_prompt, api_base, model_name, max_workers):
    """
    Main function, parameter changed from output_path to output_dir
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    user_prompt_template = '''
    Input:
    Question: {QUESTION}
    Answer: {ANSWER}
    RESPONSE: 
    '''

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, item in enumerate(data):
            api_key = API_KEYS[idx % len(API_KEYS)]
            # Pass output_dir to process_item
            futures.append(executor.submit(
                process_item, item, api_key, api_base, model_name, output_dir,
                system_prompt, user_prompt_template
            ))

        # Progress bar (no change)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc="Processing items", unit="item"):
            try:
                result = future.result()
                # You can uncomment the line below to see the processing result of each item in real-time
                # tqdm.write(result) 
            except Exception as e:
                tqdm.write(f"A task generated an exception: {e}")


if __name__ == '__main__':
    
    # --- Prompts ---
    QAC_system_prompt_no_shot = '''
You are an expert evaluator. You will be given a question and its corresponding answer. Your task has two steps:

Step 1: Classification
Determine whether the QA pair fits the Logical Reasoning category:
    •   Requires inference, comparison, or multi-step logic
    •   Examples: “Why”, “How”, “What if” questions
    •   May involve abstract thinking, causality, or analogies
If unsuitable, output:
RESPONSE: No. (and stop here)

Step 2: Scoring
If suitable, assess the quality of the answer with respect to the following aspects:
    •   Correctness: Does the answer stay accurate and aligned with the question?
    •   Completeness: Does it address all relevant points without major gaps?
    •   Clarity: Is the explanation easy to follow, concise, and unambiguous?
    •   Instruction & Format Compliance: Does it follow the intended task style and structure?
    •   Reasoning Quality: Does it show logical, coherent, step-by-step reasoning (not just surface-level claims)?

Give a binary score:
    •   Score 2: The answer is correct, clear, and mostly complete. It demonstrates good reasoning depth (multi-step logic, analogies, or thoughtful analysis) and follows the instruction well.
    •   Score 1: The answer is generally valid but weaker in one or more aspects above (e.g., shallow reasoning, missing detail, unclear expression, or partial compliance).

Also consider: Is the reasoning process made explicit in a way that helps a model learn step-by-step reasoning?

Output Format:
RESPONSE:
Score: {1 or 2}
Justification: {no more than 100 words; mention strengths and weaknesses in terms of correctness, completeness, clarity, compliance, and reasoning quality}
'''


    AVAILABLE_PROMPTS = {
        'qac_no_shot': QAC_system_prompt_no_shot,
    }

    # --- Argparse setup (modified) ---
    parser = argparse.ArgumentParser(description="Process instruction data using an LLM API.")
    
    parser.add_argument( '--input_path', type=str,  required=True,  help='Path to the input JSON file.')
    # --- Change: output_path -> output_dir ---
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for high/low score files.')
    # --- (Other parameters remain unchanged) ---
    parser.add_argument('--api_base', type=str, default="http://0.0.0.0:9000/v1", help='The base URL for the OpenAI compatible API.')
    parser.add_argument('--model_name', type=str, default='deepseek-chat',help='The name of the model to use.')
    parser.add_argument('--prompt_name', type=str, default='qac_no_shot', choices=AVAILABLE_PROMPTS.keys(), help='The name of the system prompt to use.')
    parser.add_argument('--max_workers', type=int, default=len(API_KEYS), help='Maximum number of concurrent threads.')
    parser.add_argument("--name", type=str, required=True, help="Base name for output files (e.g., 'Dolly', 'Alpaca').")

    args = parser.parse_args()

    # --- Run main program (modified) ---

    start_time = time.time() 

    selected_system_prompt = AVAILABLE_PROMPTS[args.prompt_name]

    # --- Change: Handle output directory and files ---
    output_file_high = os.path.join(args.output_dir, f"{args.name}_stage_1_high.json")
    output_file_low = os.path.join(args.output_dir, f"{args.name}_stage_1_low.json")

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    if os.path.exists(output_file_high):
        print(f"Warning: Output file {output_file_high} already exists and will be overwritten.")
        os.remove(output_file_high)

    if os.path.exists(output_file_low):
        print(f"Warning: Output file {output_file_low} already exists and will be overwritten.")
        os.remove(output_file_low)

    print(f"Starting processing...")
    print(f"Input file: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"API Base: {args.api_base}")
    print(f"Model: {args.model_name}")
    print(f"Workers: {args.max_workers}")
    print(f"Prompt: {args.prompt_name}")

    main(
        input_path=args.input_path, 
        output_dir=args.output_dir,
        system_prompt=selected_system_prompt,
        api_base=args.api_base,
        model_name=args.model_name,
        max_workers=args.max_workers
    )

    end_time = time.time()
    total_time = end_time - start_time

    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\nProcessing complete. Results saved to {args.output_dir}")
    print(f"Total execution time: {minutes} minutes and {seconds} seconds.")

