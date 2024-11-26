import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import os
import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')


def emb_gen(input_tensor, model):
    if input_tensor is None:
        return ''

    # 提取最后一层的嵌入
    with torch.no_grad():
        output = model(input_tensor, output_hidden_states=True)
        hidden_state = output.hidden_states
        embedding = hidden_state[-1][:, -1, :] # [-1]是33层transformer的最后一层，[:, -1, :]对应最后一个token[batchsize，token，hidden size]

    return embedding


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained("/home/llama3-8b-instruct").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/home/llama3-8b-instruct")
    tokenizer.pad_token = tokenizer.eos_token

    with open('/home/exp_data/toys_ctr_data.pkl', 'rb') as file:
        datas = pickle.load(file)
        datas['emb'] = [None]*1418355
        
        prompt_batch = 256
        emb_list = []
        for batch_num in range(0,1418355,prompt_batch):
            prompt_list = datas['Prompt'][batch_num:(batch_num+prompt_batch)]
            input_ids_list = []

            for prompt in prompt_list:
                input_ids = tokenizer.encode(
                    prompt, 
                    add_special_tokens=True, 
                    padding='max_length',
                    max_length=128,)
                input_ids = input_ids[-128:]
                input_ids_list.append(input_ids)

            input_tensor = torch.tensor(input_ids_list).to(device)
            emb = emb_gen(input_tensor, model)

        
            tensors = emb.cpu().numpy()
            # tensors_r = np.round(tensors, decimals=4)

            for j in range(prompt_batch):
                datas['emb'][(batch_num+j)] = tensors[:][j]

                
            with open(f'/home/exp_data/toys_ctr_data_emb_epoch0_{batch_num+prompt_batch}_layer-1_token-1.pkl', 'wb') as resfile:
                pickle.dump(datas, resfile)

            last_file = f'/home/exp_data/toys_ctr_data_emb_epoch0_{batch_num}_layer-1_token-1.pkl'
            if os.path.exists(last_file):
                os.remove(last_file)


        

if __name__=="__main__":
    run()

    # with open('/home/exp_data/toys_ctr_data.pkl', 'rb') as file:
    #     datas = pickle.load(file)
    #     print(datas['Prompt'][0])
    #     print(len(datas['Prompt']))