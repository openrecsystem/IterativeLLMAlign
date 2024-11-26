import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import os
import pandas as pd
import numpy as np
import warnings
import time
from copy import deepcopy
import argparse
from datetime import datetime

# 获取当前时间
current_time = datetime.now()
s_time = current_time.strftime('%Y%m%d%H%M%S')
print("current time:", s_time)

warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "16"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def emb_gen(input_tensor, model):
    if input_tensor is None:
        return ''

    # 提取最后一层的嵌入
    with torch.no_grad():
        output = model(input_tensor, output_hidden_states=True)
        hidden_state = output.hidden_states
        embedding = hidden_state[-1][:, -1, :] # [-1]是33层transformer的最后一层，[:, -1, :]对应最后一个token[batchsize，token，hidden size]

    return embedding

def array_to_string(arr):
    if arr is None:
        return None
    elif isinstance(arr, str):
        return arr
    else:
        return ','.join(map(str, arr))


def embedding():
    parser = argparse.ArgumentParser(description='处理模型文件')
    parser.add_argument('--model', type=str, default="/mnt/data/0/xuchao/llama3-8b-instruct", help='模型文件的路径')
    parser.add_argument('--prompt_file', type=str, default='/mnt/data/0/xuchao/exp_data/sample_data_with_prompt.csv', help='模型文件的路')
    parser.add_argument('--emb_save_file', type=str, default='/mnt/data/0/xuchao/exp_data/sampled_ml_1m_emb_epoch0.csv', help='模型文件的路径')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.prompt_file, 'rb') as file:
        # datas = pickle.load(file)
        datas = pd.read_csv(file)
        data_L = len(datas['prompt']) #739012
        # print(f"data length: {data_L}")
        datas['emb'] = [None]*data_L
        
        prompt_batch = 64
        end_point = (data_L // prompt_batch)*prompt_batch
        for batch_num in range(0,end_point,prompt_batch):
            prompt_list = datas['prompt'][batch_num:(batch_num+prompt_batch)]
            input_ids_list = []

            for prompt in prompt_list:
                input_ids = tokenizer.encode(
                    prompt, 
                    add_special_tokens=True, 
                    padding='max_length',
                    max_length=256,)
                input_ids = input_ids[-256:]
                input_ids_list.append(input_ids)

            input_tensor = torch.tensor(input_ids_list).to(device)
            emb = emb_gen(input_tensor, model)
            tensors = emb.cpu().numpy()
            tensors_r = np.round(tensors, decimals=4)

            for j in range(prompt_batch):
                datas['emb'][(batch_num+j)] = tensors_r[:][j]

            with open(f'/mnt/data/0/xuchao/llm_rec/exp_data/start_time_{s_time}_batch_{batch_num+prompt_batch}.csv', 'wb') as resfile:
                # pickle.dump(datas, resfile)
                datas['emb'] = datas['emb'].apply(array_to_string)
                datas_copy = deepcopy(datas)
                datas_copy.fillna('', inplace=True)
                # print(datas_copy['emb'])
                datas_copy.to_csv(resfile, index=False)

            last_file = f'/mnt/data/0/xuchao/llm_rec/exp_data/start_time_{s_time}_batch_{batch_num-prompt_batch}.csv'
            if os.path.exists(last_file):
                os.remove(last_file)

        ## processing the rest
        prompt_list = datas['prompt'][end_point:data_L]
        input_ids_list = []

        for prompt in prompt_list:
            input_ids = tokenizer.encode(
                prompt, 
                add_special_tokens=True, 
                padding='max_length',
                max_length=256,)
            input_ids = input_ids[-256:]
            input_ids_list.append(input_ids)

        input_tensor = torch.tensor(input_ids_list).to(device)
        emb = emb_gen(input_tensor, model)
        tensors = emb.cpu().numpy()
        tensors_r = np.round(tensors, decimals=4)

        for j in range(end_point,data_L):
            datas['emb'][j] = tensors_r[:][(j%end_point)]

        with open(args.emb_save_file, 'wb') as resfile:
            # print(datas['emb'])
            # pickle.dump(datas, resfile)
            datas['emb'] = datas['emb'].apply(array_to_string)
            datas_copy = deepcopy(datas)
            datas_copy.fillna('', inplace=True)
            datas_copy.to_csv(resfile, index=False)
        

if __name__=="__main__":
    embedding()
