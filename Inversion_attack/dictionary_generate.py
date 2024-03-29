from transformers import AutoModel, AutoTokenizer
import numpy as np
import random
import os
import pickle
import argparse
import re

def replace_numbers_with_zero(input_string):
    pattern = re.compile(r'\d+')
    result_string = re.sub(pattern, '0', input_string)
    return result_string

def parse_options():
    parser = argparse.ArgumentParser(description='Generate and split source code dictionary dataset.')
    parser.add_argument('-i', '--input', help='The path of normalized source code.', required=True)
    parser.add_argument('-o', '--out', help='The path of output dictionary.', required=True)
    parser.add_argument('-n', '--line',type=int, help='The number of embedding lines.', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_options()
    code_line =args.line
    seva_dict = args.out
    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    codes = []
    code_path = args.input+ "No-Vul"
    for file_name in os.listdir(code_path):
        f = open(code_path+'/'+file_name, 'r')
        data = f.readlines()
        f.close()
        for i in range(len(data)-code_line+1):
            codes.append(' '.join(data[i:i+code_line]))
            i+=code_line-1

    code_path = args.input+ "Vul"
    for file_name in os.listdir(code_path):
        f = open(code_path+'/'+file_name, 'r')
        data = f.readlines()
        f.close()
        for i in range(len(data)-code_line+1):
            codes.append(' '.join(data[i:i+code_line]))
            i+=code_line-1

    random.seed(1)
    random.shuffle(codes)

    train_size = int(0.8*len(codes))
    train_codes = codes[:int(len(codes)*0.8)]
    test_codes = codes[int(len(codes)*0.8):]

    # eliminate numbers
    for i in range(len(train_codes)):
        train_codes[i]=replace_numbers_with_zero(train_codes[i])
        train_codes[i]=train_codes[i].replace("'","\"")

    train_codes=list(set(train_codes))


    os.makedirs(seva_dict, exist_ok=True)
    os.makedirs(seva_dict+'/dict', exist_ok=True)
    os.makedirs(seva_dict+'/test', exist_ok=True)

    for i in range(len(train_codes)):
        f = open(seva_dict +"/dict/code_"+str(i)+".txt","w")
        f.write(train_codes[i])
        f.close()
        inputs= tokenizer.encode(train_codes[i], return_tensors="pt").to(device)
        embedding= model(inputs)[0]
        f = open(seva_dict +"/dict/embedding_"+str(i)+".pkl","wb")
        pickle.dump(embedding, f)
        f.close()


    for i in range(len(test_codes)):
        f = open(seva_dict +"/test/code_"+str(i)+".txt","w")
        f.write(test_codes[i].replace("'","\""))
        f.close()
        inputs= tokenizer.encode(test_codes[i].replace("'","\""), return_tensors="pt").to(device)
        embedding= model(inputs)[0]
        f = open(seva_dict +"/test/embedding_"+str(i)+".pkl","wb")
        pickle.dump(embedding, f)
        f.close()

if __name__ == "__main__":
    main()
