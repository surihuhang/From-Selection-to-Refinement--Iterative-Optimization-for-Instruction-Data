import json
import re
import os

def extract_field(text, field):
    if field == "output_new":

        pattern = r"output_new:\s*([\s\S]*)"
    else:
        raise ValueError("Only 'output_new'")
    
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None



def step6_transform_json(output_new_response):

    output = extract_field(output_new_response, "output_new")
    if not output:
        output = output_new_response
    return output
