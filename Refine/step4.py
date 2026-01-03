import openai
import os
api_key = "fake-key1"

API_BASE = os.getenv("Backbone_API_BASE", "")
MODEL_NAME = os.getenv("Backbone_MODEL_NAME", "")
TEMPERATURE = 0.6
MAX_TOKENS = 2000
TIMEOUT = 15
MAX_RETRIES = 3

def chat_with_llama(api_key, user_prompt, system_prompt="You are a helpful assistant."):
    openai.api_base = API_BASE
    openai.api_key = api_key
    for attempt in range(1, MAX_RETRIES + 1):
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


def step4_process_item(new_instruction, new_input):
    """
    For a single example:
    - Concatenate instruction + input
    - Call the model to get the answer
    - Write the answer to item['output_model']
    - Write the target JSONL file line by line
    """
    global counter

    instr = new_instruction
    inp = new_input
    
    if inp:
        user_prompt = f"{instr}\n{inp}"
    else:
        user_prompt = instr
    response = chat_with_llama(api_key, user_prompt)

    return response