import json
import argparse
import os

def process(in_1_path, in_2_path, out_path):
    """
    Loads two JSON files, removes specific keys from the first file's data,
    merges the data, and saves to an output file.
    """
    print(f"Loading data from: {in_1_path}")
    with open(in_1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    print(f"Loading data from: {in_2_path}")
    with open(in_2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # Traverse and remove specified keys from the first dataset
    print("Processing data from first file (removing score, IFD_Score, Class, label)...")
    keys_to_remove = ['score', 'IFD_Score', 'Class', 'label']
    for item in data1:
        for key in keys_to_remove:
            item.pop(key, None)

    # Combine the two datasets
    data = data1 + data2
    print(f"Total items combined: {len(data)}")

    # Ensure output directory exists
    output_dir = os.path.dirname(out_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the combined data to the output file
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully saved combined data to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge two JSON files, cleaning keys from the first file.")
    
    parser.add_argument('--input_path_1', type=str, required=True, help='Path to the first input JSON file (keys will be cleaned).')
    parser.add_argument('--input_path_2', type=str, required=True, help='Path to the second input JSON file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the merged output JSON file.')
    
    args = parser.parse_args()
    
    process(args.input_path_1, args.input_path_2, args.output_path)

if __name__ == "__main__":
    main()
