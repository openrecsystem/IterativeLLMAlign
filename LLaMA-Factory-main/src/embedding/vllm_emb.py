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

# 获取当前时间
current_time = datetime.now()
s_time = current_time.strftime('%Y%m%d%H%M%S')
print("current time:", s_time)

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

# def _create_model_runner(model: str, *args, **kwargs) -> ModelRunner:
#     engine_args = EngineArgs(model, *args, **kwargs)
#     engine_config = engine_args.create_engine_config()
#     model_runner = ModelRunner(
#         model_config=engine_config.model_config,
#         parallel_config=engine_config.parallel_config,
#         scheduler_config=engine_config.scheduler_config,
#         device_config=engine_config.device_config,
#         cache_config=engine_config.cache_config,
#         lora_config=engine_config.lora_config,
#         load_config=engine_config.load_config,
#         is_driver_worker=True
#     )
#     return model_runner

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
    # model = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.9) 
    model = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.9, enforce_eager=True)

    with open(args.prompt_file, 'rb') as file:
        datas = pickle.load(file)
        # datas = pd.read_csv(file)

    data_L = len(datas['prompt']) #739012
    datas['emb'] = [None]*data_L
        
    prompt_batch = args.batch_size

    for batch_num in range(0, data_L, prompt_batch):

        prompts = datas['prompt'][batch_num:batch_num + prompt_batch].tolist()
        
        outputs = model.encode(prompts=prompts)

        embeddings = []
        for output in outputs:
            last_token_emb = output.outputs.embedding
            embeddings.append(last_token_emb)
            # print( "emb:", last_token_emb)
        embeddings = np.array(embeddings)

        # embeddings = torch.stack(embeddings)
        print("embeddings: ", embeddings.shape)

        # tensors = emb.cpu().numpy()
        tensors_r = np.round(embeddings, decimals=4)


        for j in range(tensors_r.shape[0]):
            datas['emb'][(batch_num+j)] = tensors_r[:][j]

        cur_num = min(batch_num+prompt_batch, data_L)

        print(f"complated {cur_num}")
        
        # with open(f'/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/start_time_{s_time}_batch_{cur_num}.pkl', 'wb') as resfile:
        #     pickle.dump(datas, resfile)
        
    with open(args.emb_save_file, 'wb') as resfile:
        pickle.dump(datas, resfile)
               
    trans_llm_architecture(model_config_path, SOURCE_ARCHITECTURE)

if __name__=="__main__":
    embedding()
