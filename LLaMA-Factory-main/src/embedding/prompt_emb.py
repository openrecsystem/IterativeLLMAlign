import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import os

os.environ["OMP_NUM_THREADS"] = "16"

device = "cuda:3"
model = AutoModelForCausalLM.from_pretrained("/cpfs/29a75185021b187f/mistral")
tokenizer = AutoTokenizer.from_pretrained("/cpfs/29a75185021b187f/mistral")
model.to(torch.device(device))

def emb_gen(row):
    text = row.get('prompt')
    if not pd.notnull(text):
        return ''

    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids]).to(torch.device(device))

    # 提取最后一层的嵌入
    with torch.no_grad():
        output = model(input_tensor, output_hidden_states=True)
        hidden_state = output.hidden_states
        embedding = hidden_state[-2][:, -1, :] # [-1]是33层transformer的最后一层，[:, -1, :]对应最后一个token[batchsize，token，hidden size]

    emb_np = embedding.cpu().numpy()
    emb_list = emb_np.tolist()[0]
    emb_list = [str(round(e, 4)) for e in emb_list]
    return ','.join(emb_list)

def run():
    with open('embs20240117L2.csv', 'w', newline='', encoding='utf-8') as csvfile:
        sampleNo = 0
        all_emb = []
        chunksize = 1

        for df_chunk in pd.read_csv('prompts20240117.csv', sep=',', chunksize=chunksize):
            df_chunk['emb'] = df_chunk.apply(emb_gen, axis=1)
            df1_chunk = df_chunk[['user_id', 'item_id', 'p_hour', 'prompt', 'emb']]
            df1_chunk.to_csv(csvfile, mode='a', header=(sampleNo == 0), index=False)
            sampleNo += 1
            if sampleNo % 100 == 0:
                current_time = datetime.now()
                print('current_time:', current_time, '   sample No:', sampleNo)


if __name__=="__main__":
    run()



