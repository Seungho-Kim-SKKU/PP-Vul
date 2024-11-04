from transformers import AutoModel, AutoTokenizer
import numpy as np
import random
import os
import pickle
from time import time
import re
import argparse

def replace_numbers_with_zero(input_string):
    pattern = re.compile(r'\d+')
    result_string = re.sub(pattern, '0', input_string)
    return result_string

def Cosine_similarity(a,b):
    sim = 0
    sum_a = 0
    sum_b = 0
    for i in range(len(a)):
        sum_a += (a[i])**2
        sum_b += (b[i])**2
        sim += a[i]*b[i]
    return sim/(sum_a*sum_b)**(1/2)

def parse_options():
    parser = argparse.ArgumentParser(description='Generate and split source code dictionary dataset.')
    parser.add_argument('-i', '--input', help='The path of normalized source code.', required=True)
    parser.add_argument('-n', '--line',type=int, help='The number of embedding lines.', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_options()
    device = "cuda"  # for GPU usage or "cpu" for CPU usage
    dict_codes = []
    dict_embeddings = []
    code_line =args.line
    dict_path = args.input + '/dict'
    test_path = args.input + '/test'


    dict_length = int(len(os.listdir(dict_path))/2)
    for i in range(dict_length):
        f = open(dict_path+'/code_'+str(i)+'.txt', 'r')
        data = f.readlines()
        f.close()
        dict_codes.append(' '.join(data[0:code_line]))

    for i in range(dict_length):
        f = open(dict_path+'/embedding_'+str(i)+'.pkl', 'rb')
        data = pickle.load(f)
        f.close()
        dict_embeddings.append(data)
            
    average_time =0
    accuracy = 0
    codes = []
    embeddings = []
    test_length = int(len(os.listdir(test_path))/2)
    for i in range(test_length):
        f = open(test_path+"/code_"+str(i)+".txt", 'r')
        codes = f.readlines()
        f.close()
        f = open(test_path+"/embedding_"+str(i)+".pkl", 'rb')
        data = pickle.load(f)
        f.close()
        max = 0
        expect = 0
        start = time()
        for j in range(len(dict_embeddings)):
            similarity = Cosine_similarity(data,dict_embeddings[j])
            if similarity > max:
                expect = j
                max = similarity
        end = time()
        average_time+= end - start
        if replace_numbers_with_zero(' '.join(codes[0:code_line])) == dict_codes[expect]:
            accuracy+=1
        else:
            print("original:\n",' '.join(codes[0:code_line]))
            print("predict:\n",dict_codes[expect])
            print("similarity:", max)

    print('average_time:',average_time/test_length)
    print('accuracy:',accuracy/test_length)

if __name__ == "__main__":
    main()
