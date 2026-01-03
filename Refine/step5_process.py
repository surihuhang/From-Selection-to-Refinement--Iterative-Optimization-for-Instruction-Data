import json
import re
import os

def extract_field(text, field):
    match = re.search(r'\{\s*"winner"\s*:\s*.*?"explanation"\s*:\s*.*?\}', text, re.DOTALL)
    if not match:
        return None
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        json_candidate = json_str.strip()
        try:
            data = json.loads(json_candidate)
        except Exception:
            return None
    if field not in ("winner", "explanation"):
        raise ValueError("Only 'winner' and 'explanation'")

    return data.get(field)



def step5_process_transform_json(modified_text):
    winner = extract_field(modified_text, "winner")
    explanation = extract_field(modified_text, "explanation")
    if winner == 'A':
        win = True
    else:
        win = False

    return win, explanation
