from vllm import ModelRegistry
from llama_for_embedding import LlamaEmbeddingModel
ModelRegistry.register_model("LlamaEmbModel", LlamaEmbeddingModel)

from vllm.model_executor.models import _EMBEDDING_MODELS
global _EMBEDDING_MODELS
_EMBEDDING_MODELS["LlamaEmbModel"] = LlamaEmbeddingModel


from cgitb import enable
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
from vllm import LLM, SamplingParams
from vllm.worker.model_runner import ModelRunner
from vllm.engine.arg_utils import EngineArgs
import json
import psutil

# 获取当前时间
current_time = datetime.now()
s_time = current_time.strftime('%Y%m%d%H%M%S')
print("current time:", s_time)

pid = os.getpid()
process = psutil.Process(pid)


warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "16"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SOURCE_ARCHITECTURE = ["LlamaForCausalLM"]
TARGET_ARCHITECTURE = ["LlamaEmbModel"]


def trans_llm_architecture(config_path, architecture):
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["architectures"] = architecture

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


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
    parser.add_argument('--model', type=str, default="/mnt/data/0/LLM4Rec/llm_models/llama3-8b-instruct", help='模型文件的路径')
    parser.add_argument('--prompt_file', type=str, default='/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data_all/sample_data_with_prompt.pkl', help='模型文件的路')
    parser.add_argument('--emb_save_file', type=str, default='/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data_all/all_ml_1m_emb_epoch0.pkl', help='模型文件的路径')
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size")
    args = parser.parse_args()

    model_config_path = f"{args.model}/config.json"
    trans_llm_architecture(model_config_path, TARGET_ARCHITECTURE)

    # model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1, seed=0)
    model = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.9, enforce_eager=True)

    with open(args.prompt_file, 'rb') as file:
        datas = pickle.load(file)
        # datas = datas.head(10)

    data_L = len(datas['prompt'])
    datas['emb'] = [None]*data_L
        
    
    prompts = datas['prompt'].tolist()
    mem = process.memory_info().rss
    print(f"before emb memory: {mem / (1024*1024*1024):.2f} G")

    outputs = model.encode(prompts=prompts)
    embeddings = [output.outputs.embedding for output in outputs]
    embeddings = np.round(np.array(embeddings), decimals=7)

    mem = process.memory_info().rss
    print(f"after emb memory: {mem / (1024*1024*1024):.2f} G")
    for i in range(data_L):
        datas['emb'][i] = np.array(embeddings[:][i])

    mem = process.memory_info().rss
    print(f"before save memory: {mem / (1024*1024*1024):.2f} G")
        
    
    save_columns = ['User ID', 'Movie ID','emb']
    if 'toys' in args.emb_save_file:
        save_columns[1] = 'Item ID'
    if 'bookcross' in args.emb_save_file:
        save_columns[1] = 'ISBN'
    with open(args.emb_save_file, 'wb') as resfile:
        pickle.dump(datas[save_columns], resfile)
    
    mem = process.memory_info().rss
    print(f"after save memory: {mem / (1024*1024*1024):.2f} G")
               
    trans_llm_architecture(model_config_path, SOURCE_ARCHITECTURE)

if __name__=="__main__":
    embedding()

    # emb_file='/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data_all/all_ml_1m_emb_epoch0.pkl'
    # with open(emb_file, 'rb') as file:
    #     datas = pickle.load(file)
    #     print(datas)
    #     print(datas['emb'][0])
