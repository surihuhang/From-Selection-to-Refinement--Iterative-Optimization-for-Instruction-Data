import os
import json
import torch
import argparse
from tqdm import tqdm
import time
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn as nn
import math

# --- Global definitions (from data_analysis.py) ---
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# --- Helper functions (from data_analysis.py) ---

def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad(): 
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)
    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)
    return perplexity.to('cpu'), sentence_embedding.to('cpu')

def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]
    labels = input_ids.clone()
    labels[0, :start_token] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    perplexity = torch.exp(loss)
    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())
    return perplexity.to('cpu'), 0, losses

# --- Helper functions (from data_by_cluster_Test.py) ---

def do_clustering(args, high_dim_vectors):
    clustering_algorithm = args.cluster_method
    if clustering_algorithm == 'kmeans':
        print(f"Running KMeans clustering with {args.kmeans_num_clusters} clusters...")
        clustering = KMeans(n_clusters=args.kmeans_num_clusters, random_state=0, n_init='auto').fit(high_dim_vectors)
        print("Clustering complete.")
    else:
        raise NotImplementedError(f"Clustering method '{clustering_algorithm}' is not supported.")
    return clustering

# --- Helper functions (from your new script) ---

def process_and_label_data_proportional(data, k_ratio):
    """
    Process JSON data, partition, group, sort, select, and label based on the specified [ratio k_ratio].
    (Full copy of the function you provided)
    """
    if not isinstance(k_ratio, (int, float)) or not (0.0 <= k_ratio <= 1.0):
        raise ValueError(f"k_ratio must be a float or int between 0.0 and 1.0. Current value: {k_ratio}")

    # 1. Data partitioning: Partition data based on 'Class'
    partitioned_data = {}
    class_ids = set() # Used to count the number of classes
    for item in data:
        class_id = item.get("Class")
        class_ids.add(class_id)
        if class_id not in partitioned_data:
            partitioned_data[class_id] = []
        partitioned_data[class_id].append(item) 

    num_classes = len(class_ids)
    if num_classes == 0:
        print("Warning: No classes found in the data.")
        return {"_overall_summary": {"total_classes": 0, "k_ratio": k_ratio, "total_actual_selected": 0}}, data

    print(f"Detected {num_classes} classes. Will use ratio k_ratio = {k_ratio} (i.e., {k_ratio:.0%}) for data filtering.\n")

    selected_items_ids = set()
    selection_details = {}
    total_actual_selected = 0

    for class_id, items in partitioned_data.items():
        total_items_in_class = len(items)
        current_class_target_k = int(round(total_items_in_class * k_ratio))
        
        # 2. Data grouping (using the 'score' field from your example)
        group_1 = [item for item in items if item.get("score") == 1]
        group_2 = [item for item in items if item.get("score") == 2]

        high_score_data_count = len(group_2)
        low_score_data_count = len(group_1)

        # 3. Data sorting
        group_1_sorted = sorted(group_1, key=lambda x: x.get("IFD_Score", 0), reverse=True)
        group_2_sorted = sorted(group_2, key=lambda x: x.get("IFD_Score", 0), reverse=True)

        # 4. Data selection
        top_k_for_class = []
        
        take_from_group_2 = min(current_class_target_k, len(group_2_sorted)) 
        top_k_for_class.extend(group_2_sorted[:take_from_group_2])
        
        remaining_k_needed = current_class_target_k - len(top_k_for_class)
        if remaining_k_needed > 0:
            take_from_group_1 = min(remaining_k_needed, len(group_1_sorted))
            top_k_for_class.extend(group_1_sorted[:take_from_group_1])
            
        actual_selected_count_for_class = len(top_k_for_class)
        
        selection_details[class_id] = {
            "total_items_in_class": total_items_in_class,
            "high_score_data_count": high_score_data_count,
            "low_score_data_count": low_score_data_count,
            "target_k_for_class": current_class_target_k,
            "actual_selected_count": actual_selected_count_for_class
        }
        total_actual_selected += actual_selected_count_for_class
        
        for selected_item in top_k_for_class:
            selected_items_ids.add(id(selected_item))

    # 5. Add "label" field
    labeled_data = []
    all_original_items = []
    for items_in_class in partitioned_data.values():
        all_original_items.extend(items_in_class)

    for item in all_original_items:
        if id(item) in selected_items_ids:
            item["label"] = "High"
        else:
            item["label"] = "Low"
        labeled_data.append(item)

    selection_details["_overall_summary"] = {
        "total_classes": num_classes,
        "k_ratio": k_ratio,
        "total_actual_selected": total_actual_selected
    }

    return selection_details, labeled_data


# --- Merged Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline: Calculate PPL/Emb, Cluster, and Filter by k_ratio.")
    
    # --- Part 1 (Calc) ---
    parser.add_argument("--json_data_path", type=str, required=True, help="Path to the original JSONL data file (must contain 'score' field).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='alpaca', help='wiz, alpaca')
    parser.add_argument("--mod", type=str, default='pre', help='pre, cherry')

    # --- Part 2 (Cluster) ---
    parser.add_argument("--sent_type", type=int, default=0, help="Index of the sentence embedding to use.")
    parser.add_argument("--ppl_type", type=int, default=0, help="Index of the PPL/IFD score to use.")
    parser.add_argument("--cluster_method", type=str, default='kmeans', help="Clustering method to use (default: kmeans).")
    parser.add_argument("--kmeans_num_clusters", type=int, default=132, help="Number of clusters for KMeans.")
    
    # --- Part 3 (Filter & Save) ---
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save final 'high' and 'low' JSON files.")
    parser.add_argument("--k_ratio", type=float, default=0.8, help="Proportion (0.0-1.0) of data to select as 'High' from each cluster.")
    parser.add_argument("--name", type=str, required=True, help="Base name for output files (e.g., 'Dolly', 'Alpaca').")

    args = parser.parse_args()
    return args

# --- Merged Main Function ---

def main():
    args = parse_args()
    print("Full Pipeline: Analysis, Clustering, and Proportional Selection.")
    print(args)
    input_json = os.path.join(args.json_data_path , "Data_stage_1_all.json")

    # =========================================================================
    # PART 1: Calculate PPL and Embeddings
    # =========================================================================
    
    print("PART 1: Loading model and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        cache_dir="../cache"
    )
    model.config.output_hidden_states = True
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir="../cache",
        use_fast=True,
        legacy=False
    )

    # --- Fix: Load using JSONL format ---
    print(f"Loading data from {input_json} (JSONL format)...")
    full_json_data = []
    with open(input_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): # Avoid empty lines
                try:
                    full_json_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    # --- End of fix ---

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(full_json_data)
    sampled_json_data = full_json_data[start_idx:end_idx] 

    print(f"Processing {len(sampled_json_data)} items from index {start_idx} to {end_idx}...")
    
    start_time_calc = time.time()
    pt_data_in_memory = [] # In-memory .pt content

    for i in tqdm(range(len(sampled_json_data)), desc="PART 1: Calculating PPL/Embeddings"):
        data_i = sampled_json_data[i]
        
        # Confirm data format (based on your example)
        instruct_i = data_i['instruction']
        output_i = data_i['output']
        input_i = data_i.get('input', '') # Safely get input

        direct_answer_text = '### Response:' + output_i
        
        if args.prompt == 'wiz':
            whole_text = instruct_i+'\n\n### Response:'+output_i
            if input_i != '':
                whole_text = instruct_i+'\nInput:'+input_i+'\n\n### Response:'+output_i
        
        elif args.prompt == 'alpaca':
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

        temp_data_i = {}
        if args.mod == 'pre':
            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, instruct_i, args.max_length)
            temp_data_i['ppl'] = [ppl_ins_alone, 0, 0]
            temp_data_i['sent_emb'] = [emb_ins_alone, 0, 0]
        elif args.mod == 'cherry':
            instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            instruct_i_len = instruct_i_input_ids.shape[1] 
            ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_part_text(tokenizer, model, direct_answer_text, output_i, args.max_length-instruct_i_len+4)
            ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, args.max_length)
            temp_data_i['ppl'] = [0, ppl_out_alone, ppl_out_condition]
            temp_data_i['token_loss'] = [[], loss_list_alone, loss_list_condition]

        pt_data_in_memory.append(temp_data_i)
    
    print(f'PART 1 complete. Time Used: {(time.time()-start_time_calc)/60:.2f} (min)')
    
    print("Releasing model from memory...")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # PART 2: Clustering and Data Merging
    # =========================================================================
    print("PART 2: Starting Clustering and Data Merging...")
    
    pt_data = pt_data_in_memory
    json_data_to_update = sampled_json_data

    if len(pt_data) != len(json_data_to_update):
        raise ValueError(f"Data length mismatch: In-memory .pt data ({len(pt_data)}) vs .json data ({len(json_data_to_update)}).")

    print("Extracting embeddings and scores for clustering...")
    emb_list = []
    ifd_score_list = []
    for i in tqdm(range(len(pt_data)), desc="PART 2: Extracting"):
        data_i = pt_data[i]
        emb_list.append(data_i['sent_emb'][args.sent_type])
        ifd_score_list.append(data_i['ppl'][args.ppl_type].item())

    high_dim_vectors = torch.cat(emb_list, 0).numpy()
    ifd_scores = np.array(ifd_score_list)

    clustering = do_clustering(args, high_dim_vectors)
    cluster_labels = clustering.labels_

    print("Merging scores and cluster labels into JSON data...")
    updated_data = [] # In-memory (json + IFD_Score + Class)
    for i in tqdm(range(len(json_data_to_update)), desc="PART 2: Updating JSON"):
        data_entry = json_data_to_update[i]
        data_entry['IFD_Score'] = ifd_scores[i]
        data_entry['Class'] = int(cluster_labels[i])
        updated_data.append(data_entry)
    
    print(f'PART 2 complete. {len(updated_data)} items processed.')

    # =========================================================================
    # PART 3: Filter by proportion and save High/Low
    # =========================================================================
    print("PART 3: Starting Proportional Selection and Labeling...")
    
    # Call the new function
    details, final_labeled_data = process_and_label_data_proportional(updated_data, args.k_ratio)

    print("--- Selection Details ---")
    print(json.dumps(details, indent=4, ensure_ascii=False))
    print("\n" + "="*40 + "\n")

    # Filter out High and Low data
    high_data = [item for item in final_labeled_data if item.get("label") == "High"]
    low_data = [item for item in final_labeled_data if item.get("label") == "Low"]

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct the output file paths
    k_ratio_percent = int(args.k_ratio * 100)
    
    output_high_filename = os.path.join(args.output_dir, f"{args.name}_stage_2_high.json")
    output_low_filename = os.path.join(args.output_dir, f"{args.name}_stage_2_low.json")
    result_high_filename = os.path.join(args.output_dir, f"{args.name}_high.json")
    # Save data with label "High" (as a JSON array)
    with open(output_high_filename, 'w', encoding='utf-8') as f:
        json.dump(high_data, f, ensure_ascii=False, indent=4)
    print(f"High priority data saved to: {output_high_filename} ({len(high_data)} items)")

    # Save data with label "High" (as a JSON array)

    with open(result_high_filename, 'w', encoding='utf-8') as f:
        json.dump(high_data, f, ensure_ascii=False, indent=4)
    print(f"High priority data saved to: {result_high_filename} ({len(high_data)} items)")


    # Save data with label "Low" (as a JSON array)
    with open(output_low_filename, 'w', encoding='utf-8') as f:
        json.dump(low_data, f, ensure_ascii=False, indent=4)
    print(f"Low priority data saved to: {output_low_filename} ({len(low_data)} items)")

    print(f"\nPART 3 complete. All files saved to {args.output_dir}.")


if __name__ == '__main__':
    main()
