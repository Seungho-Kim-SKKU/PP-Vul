from pp_vul import *
from he_friendly_model import CNN_Classifier, CNN, Dataset
from pp_vul.utils.structure import *

from seal import *
from torchvision import datasets
import numpy as np
import torch
from tqdm import tqdm
import sys, os
import argparse
import pickle

MAX_LEN = 50
HIDDEN_SIZE = 256

def load_data(filename):
    print("Loading data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The dir path of dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_path = parse_options().input
    data_path = data_path + "/" if data_path[-1] != "/" else data_path
    test_df = load_data(data_path + "test.pkl")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = CNN(hidden_size = HIDDEN_SIZE)
    model = torch.load(root_dir + '/model/pp-vul_16_4_x^3.pth', map_location=device)
    
    context = Context(N = 2**14, depth = 5, LogQ = 40, LogP = 60)
    embedding_size = Cuboid(1, MAX_LEN, HIDDEN_SIZE)

    HE_vul = HE_CNN(model, embedding_size, context)
    print(HE_vul)
    print('='*50)

    # num_of_data = int(context.number_of_slots // HE_vul.data_size)
    num_of_data = 1
    """
    Test dataset import part
    """ 
    X_test, y_test = test_df['data'], test_df['label']
    # test_dataset = Dataset(X_test, y_test, max_len=MAX_LEN, hidden_size=HIDDEN_SIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True)    
    
    num_of_test = 10
    
    for k in range(num_of_test):
        data = np.zeros(shape=(1, MAX_LEN, HIDDEN_SIZE))
        for j in range(1):
            for i in range(min(len(X_test[k]) - 1, MAX_LEN - 1)):
                data[j][i + 1] = X_test[k][i + 1]

        ppData = preprocessing(data, embedding_size, num_of_data, HE_vul.data_size)
        
        ciphertext_list = HE_vul.encrypt(ppData)
        
        # data1 = model.Conv1(torch.tensor(data, dtype=torch.float32))
        # print(data1[0].flatten()[:10])
        # data2 = model.Cube(data1)
        # print(data2[0].flatten()[:10])
        # data2 = model.Conv2_depthwise(data1)
        
        result = model(torch.tensor(data, dtype=torch.float32).to(device)).tolist()
        # print("Plaintext result:", result)
        # print('='*50)
    
        result_ciphertext = HE_vul(ciphertext_list, _time=True)

        result_plaintext = HE_vul.decrypt(result_ciphertext)[:2]
        result_1 = 3
        result_2 = 3
        if (result[0] > result[1]):
            result_1 = 0
        else:
            result_1 = 1
        if (result_plaintext[0] > result_plaintext[1]):
            result_2 = 0
        else:
            result_2 = 1
        # print("Plaintext result:", result_1)
        # print("Ciphertext result:", result_2)
        
        if result_1 == result_2:
            print("True")
        else:
            print("False")
    
    # print(result_plaintext[512:522])

    # for i in range(num_of_data):
    #     """Model result without homomorphic encryption"""
    #     data = torch.from_numpy(data)
    #     origin_results = model(data)[i].flatten().tolist()
    #     origin_result = origin_results.index(max(origin_results))

    #     """Model result with homomorphic encryption"""
    #     he_result = -1
    #     MIN_VALUE = -1e10
    #     sum = 0
    #     for idx in range(10):
    #         he_output = result_plaintext[idx + HE_vul.data_size*i]

    #         sum = sum + np.abs(origin_results[idx] - he_output)

    #         if(MIN_VALUE < he_output):
    #             MIN_VALUE = he_output
    #             he_result = idx

    #     """
    #     After calculating the sum of errors between the results of the original model and the model with homomorphic encryption applied, Outputting whether it matches the original results.
    #     """        
    #     print('%sth result Error: %.8f\t| Result is %s' %(str(i+1), sum, "Correct" if origin_result == he_result else "Wrong"))

        # print(i+1, 'th result')
        # print("Error          |", sum)
        # print("original label |", max_data_idx)
        # print("HE label       |", max_ctxt_idx)
        # print("real label     |", _label[i])
        # print("="*30)
