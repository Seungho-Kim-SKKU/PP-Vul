import os, sys
import glob
import argparse
import pickle
import torch
from transformers import AutoModel, AutoTokenizer

def parse_options():
    parser = argparse.ArgumentParser(description='CodeT5+ Embedding')
    parser.add_argument('-i', '--input', help='The directory path of input', type=str)
    parser.add_argument('-o', '--output', help='The directory path of output', type=str)
    parser.add_argument('-n', '--num', help= 'Embedding line', type=int)
    args = parser.parse_args()
    return args

def embedding(input_path, output_path, tokenizer, model, device, line_num):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    embeddings = []
    for i in range(0, len(lines), line_num):
        pair_of_lines = lines[i:i+line_num]
        text = ''.join(pair_of_lines)

        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            output = model(inputs)[0]
        embeddings.append(output.squeeze().cpu().numpy())

    output_file_path = input_path.replace('.c', '.pkl')
    output_file_path = output_path + output_file_path.split('/')[-1]
    
    print(output_file_path)
    
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(embeddings, output_file)

def main():
    args = parse_options()

    input_path = args.input
    output_path = args.output
    embedding_line_num = args.num
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'

    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'

    files = glob.glob(input_path + '*.c')

    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    for file_path in files:
        embedding(file_path, output_path, tokenizer, model, device, embedding_line_num)
 

if __name__ == '__main__':
    main()
