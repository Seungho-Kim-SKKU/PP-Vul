from transformers import AutoModel, AutoTokenizer
import numpy as np
import random
import os
import pickle
from time import time
import re

def replace_numbers_with_zero(input_string):
    # 정규표현식을 사용하여 숫자를 찾습니다.
    # \d는 숫자를 나타내고, +는 하나 이상의 숫자를 의미합니다.
    pattern = re.compile(r'\d+')
    
    # input_string에서 숫자를 찾아서 0으로 변경합니다.
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

device = "cuda"  # for GPU usage or "cpu" for CPU usage
dict_codes = []
dict_embeddings = []
dict_path = "/data/seonhye/PPvul/inversion/dictionary_code_line1"
dict_length = int(len(os.listdir(dict_path))/2)
for i in range(dict_length):
    f = open(dict_path+'/codes_train'+str(i)+'.txt', 'r')
    data = f.readlines()
    f.close()
    dict_codes.append(' '.join(data[0:2]))

for i in range(dict_length):
    f = open(dict_path+'/embedding_train'+str(i)+'.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    dict_embeddings.append(data)
        
average_time =0
accuracy = 0
codes = []
embeddings = []
code_path = "/data/seonhye/PPvul/inversion/split_dataset/test/codes"
embedding_path = "/data/seonhye/PPvul/inversion/split_dataset/test/embedding"
for i in range(714620,715744,1):
    f = open(code_path+"/code_"+str(i)+".txt", 'r')
    codes = f.readlines()
    f.close()
    f = open(embedding_path+"/embedding_"+str(i)+".pkl", 'rb')
    data = pickle.load(f)
    f.close()
    if i%100==0:
        print(i, "th code results")
    #print("original code:" ,codes)
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
    if replace_numbers_with_zero(' '.join(codes[0:2])).replace("'","\"") == dict_codes[expect]:
        accuracy+=1
    else:
        print("original:\n",' '.join(codes[0:2]))
        print("predict:\n",dict_codes[expect])
        print("similarity:", max)

print('average_time:',average_time/2000)
print('accuracy:',accuracy/2000)