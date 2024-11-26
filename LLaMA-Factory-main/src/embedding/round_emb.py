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


def run():
    with open('/data/gxw/LLM4Rec/data/ml-1m/ml_1m_ctr_data_emb_epoch0_738816_layer-1_token-1.pkl', 'rb') as file:
        datas = pickle.load(file)
        for j in range(738816):
            datas['emb'][j] = np.round(datas['emb'][j], decimals=1)
        print(datas['emb'][:10])

    write_file = '/data/gxw/LLM4Rec/data/ml-1m/ml_1m_ctr_data_emb_r1_epoch0_738816_layer-1_token-1.pkl'
    if not os.path.exists(write_file):
        os.mknod(write_file)
    with open(write_file, 'wb') as resfile:
        pickle.dump(datas, resfile)



if __name__=="__main__":
    run()