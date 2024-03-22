from transformers import AutoModel, AutoTokenizer
import numpy as np
import random
import os
import pickle
import argparse

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

    print(len(codes))
    train_size = int(0.8*len(codes))

    os.makedirs(seva_dict, exist_ok=True)
    os.makedirs(seva_dict +"train/codes")
    os.makedirs(seva_dict +"train/embedding")
    os.makedirs(seva_dict +"test/codes")
    os.makedirs(seva_dict +"test/embedding")

    for i in range(0,train_size,1):
        f = open(seva_dict +"train/codes/code_"+str(i)+".txt","w")
        f.write(codes[i])
        f.close()
        if i%1000==0:
            print(i,"th code save")
        
    for i in range(0,train_size,1):
        inputs= tokenizer.encode(codes[i], return_tensors="pt").to(device)
        embedding= model(inputs)[0]
        f = open(seva_dict +"train/embedding/embedding_"+str(i)+".pkl","wb")
        pickle.dump(embedding, f)
        f.close()
        if i%1000==0:
            print(i,"th embedding save")

    for i in range(train_size,len(codes),1):
        f = open(seva_dict +"test/codes/code_"+str(i)+".txt","w")
        f.write(codes[i])
        f.close()
        if i%1000==0:
            print(i,"th code save")

        
    for i in range(train_size,len(codes),1):
        inputs= tokenizer.encode(codes[i], return_tensors="pt").to(device)
        embedding= model(inputs)[0]
        f = open(seva_dict +"test/embedding/embedding_"+str(i)+".pkl","wb")
        pickle.dump(embedding, f)
        f.close()
        if i%1000==0:
            print(i,"th embedding save")

if __name__ == "__main__":
    main()
