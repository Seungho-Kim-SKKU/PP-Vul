from transformers import AutoModel, AutoTokenizer
import numpy as np
import random
import os
import pickle
from time import time
import argparse
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def Cosine_similarity(a,b):
    sim = 0
    sum_a = 0
    sum_b = 0
    for i in range(len(a)):
        sum_a += (a[i])**2
        sum_b += (b[i])**2
        sim += a[i]*b[i]
    return sim/(sum_a*sum_b)**(1/2)


parser = argparse.ArgumentParser(description='Reconstruction attack using dictionary dataset.')
parser.add_argument('-i', '--input', help='The path of dictionary dataset.', default="./dictionary")
args = parser.parse_args()


device = "cuda:0"  # for GPU usage or "cpu" for CPU usage
dict_codes = []
dict_embeddings = []
dict_path = args.input+"/dict"
dict_length = int(len(os.listdir(dict_path))/2)
for i in range(dict_length):
    f = open(dict_path+'/code_'+str(i)+'.txt', 'r')
    data = f.readlines()
    f.close()
    dict_codes.append(' '.join(data[0:2]))

for i in range(dict_length):
    f = open(dict_path+'/embedding_'+str(i)+'.pkl', 'rb')
    data = pickle.load(f).to(device)
    f.close()
    dict_embeddings.append(data)
        
average_time =0
accuracy = 0
bleu = 0
codes = []
embeddings = []
code_path = args.input+"/test"
code_length = int(len(os.listdir(code_path))/2)
for i in range(0,code_length ,1):
    f = open(code_path+"/code_"+str(i)+".txt", 'r')
    codes = f.readlines()
    f.close()
    f = open(code_path+"/embedding_"+str(i)+".pkl", 'rb')
    data = pickle.load(f).to(device)
    f.close()
    if i%100==0:
        print(i, "th code results")
    max = 0
    expect = 0
    start = time()
    for j in range(len(dict_embeddings)):
        similarity = cosine_similarity([data.cpu().detach().numpy()],[dict_embeddings[j].cpu().detach().numpy()])

        if similarity > max:
            expect = j
            max = similarity
    end = time()
    average_time+= end - start
    bleu+=sentence_bleu([codes[0][0:-1].split()],dict_codes[expect].split(), weights=(1, 0, 0, 0),smoothing_function=SmoothingFunction().method1)
    if codes[0][0:-1] == dict_codes[expect][:-1]:
        accuracy+=1
    else:
        print(codes[0][0:-1], dict_codes[expect])
print('BLEU:',bleu/100)
print('accuracy:',accuracy/100)
print("time:", average_time/100)
