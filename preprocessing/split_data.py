import os
import pickle
import random
import argparse
import pandas as pd
import sys

def read_pkl(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            print("Processing: " + filename)
            with open(os.path.join(directory, filename), "rb") as file:
                single_data = pickle.load(file)
                data.append({"filename": filename.split("/")[-1].rstrip(".pkl"),
                             "length": len(single_data),
                             "data": single_data,
                             "label": label})
    return data

def split_data(input_dir, output_dir):
    all_data = []

    for _, folder in enumerate(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            label = 0 if folder == "No-Vul" else 1
            all_data.extend(read_pkl(folder_path, label))

    all_data_df = pd.DataFrame(all_data)
    all_data_df = all_data_df.sample(frac=1).reset_index(drop=True)

    total_samples = len(all_data_df)
    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1

    train_split = int(total_samples * train_ratio)
    valid_split = int(total_samples * (train_ratio + valid_ratio))

    train_data = all_data_df[:train_split].reset_index(drop=True)
    valid_data = all_data_df[train_split:valid_split].reset_index(drop=True)
    test_data = all_data_df[valid_split:].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    def save_to_pickle(data, filename):
        with open(filename, "wb") as file:
            print("Saving: " + filename)
            pickle.dump(data, file)

    save_to_pickle(all_data, os.path.join(output_dir, "all.pkl"))
    save_to_pickle(train_data, os.path.join(output_dir, "train.pkl"))
    save_to_pickle(valid_data, os.path.join(output_dir, "valid.pkl"))
    save_to_pickle(test_data, os.path.join(output_dir, "test.pkl"))

def parse_options():
    parser = argparse.ArgumentParser(description='Generate and split train dataset.')
    parser.add_argument('-i', '--input', help='The path of input', required=True)
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_options()
    input_dir = args.input
    output_dir = args.out
    split_data(input_dir, output_dir)

if __name__ == "__main__":
    main()