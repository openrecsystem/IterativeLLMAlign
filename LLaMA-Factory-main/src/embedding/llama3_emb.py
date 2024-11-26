import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import os
import pandas as pd


os.environ["OMP_NUM_THREADS"] = "16"

model = AutoModelForCausalLM.from_pretrained("/data/xuchao/llama3-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("/data/xuchao/llama3-8b-instruct")


def emb_gen(prompt):
    if prompt is None:
        return ''

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])

    # 提取最后一层的嵌入
    with torch.no_grad():
        output = model(input_tensor, output_hidden_states=True)
        hidden_state = output.hidden_states
        embedding = hidden_state[-1][:, -1, :] # [-1]是33层transformer的最后一层，[:, -1, :]对应最后一个token[batchsize，token，hidden size]

    emb_np = embedding.cpu().numpy()
    # emb_list = emb_np.tolist()[0]
    # emb_list = [str(round(e, 4)) for e in emb_list]
    # return ','.join(emb_list)
    return emb_np[0]


def run():
    with open('/data/xuchao/exp_data/ml-1m_ctr_data.pkl', 'rb') as file:
        datas = pickle.load(file)
        datas['emb'] = [None]*739012
        sample_no = 0
        for prompt in datas['Prompt']:
            # if sample_no >= 1:
            #     break
            emb = emb_gen(prompt)
            datas['emb'][sample_no] = emb
            sample_no += 1
        
            if sample_no % 100 == 0:
                with open(f'/data/xuchao/exp_data/ml-1m_ctr_data_emb_{sample_no}.pkl', 'wb') as resfile:
                    pickle.dump(datas, resfile)

                last_file = f'/data/xuchao/exp_data/ml-1m_ctr_data_emb_{sample_no-100}.pkl'
                if os.path.exists(last_file):
                    os.remove(last_file)

if __name__=="__main__":
    run()

    # # save file
    # with open('/data/xuchao/exp_data/ml-1m_ctr_data_emb_236800.pkl', 'rb') as file:
    #     datas = pickle.load(file)
    #     print(datas['emb'])