import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from CTR.data_loader import MovieLensDataset, ToysDataset, BookcrossDataset
from CTR.ctr_model.dcnv2 import DCNV2
from CTR.trainer import Trainer
import random
import pickle
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

def init_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctr_data_path', type=str, default="CTR/data/ml-1m/proc_data/ctr_data.csv",
                        help="the path of ctr data file")
    parser.add_argument('--emb_data_path', type=str, default="CTR/data/ml-1m/proc_data/all_ml_1m_emb_epoch1.pkl",
                        help="the path of embedding data file")
    parser.add_argument('--meta_data_path', type=str, default="CTR/data/ml-1m/proc_data/ml-1m-meta.json",
                        help="the meta data path")
    parser.add_argument('--is_vec', type=bool, default=False, help="whether to use embedding vector")
    parser.add_argument('--is_hist', type=bool, default=False, help="whether to use hist")
    parser.add_argument('--vec_dim', type=int, default=0, help="the dim of embedding vector")
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size")
    parser.add_argument('--fusion_type', type=str, default="concat", help="embedding fusion method")
    parser.add_argument('--scheduler_type', type=str, default="cosine", help="learning rate scheduler type")
    parser.add_argument('--epoch', type=int, default=30, help="the max epoch")
    parser.add_argument('--model', type=str, default="dcn", help="model name")
    parser.add_argument('--early_stop', type=bool, default=False, help="early stop")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--dataset', type=str, default="ml-1m", help="dataset name")
    parser.add_argument('--pth_name', type=str, default="no_emb", help="ctr model weight path")

    args = parser.parse_args()
    init_seed()
    ctr_data_path = args.ctr_data_path
    emb_data_path = args.emb_data_path
    meta_data_path = args.meta_data_path
    is_vec = args.is_vec
    vec_dim = args.vec_dim
    batch_size = args.batch_size
    fusion_type = args.fusion_type
    scheduler_type = args.scheduler_type
    max_epoch = args.epoch
    model_name = args.model
    early_stop = args.early_stop
    data_name = args.dataset
    pth_name = args.pth_name
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emb_pkl = emb_data_path.split('.')[0].split('/')[-1]

    model_save_path = f"/mnt/data/0/LLM4Rec/CTR/data/{data_name}/ctr_model_weights/{model_name}_{pth_name}_best_model.pth"

    if not is_vec:
        emb_pkl = 'no_emb'

    log_name = f"CTR/logs/dataset_{data_name}_{model_name}_early_stop_{early_stop}_is_vec_{is_vec}_batch_size_{batch_size}_epochs_{max_epoch}_{emb_pkl}.log"

    logging.basicConfig(level=logging.INFO, filename=log_name, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    ctr_data = pd.read_csv(ctr_data_path)
    meta_data = json.load(open(meta_data_path, 'r'))

    if is_vec:
        with open(emb_data_path, 'rb') as file:
            emb_data = pickle.load(file)
        ctr_data['embed'] = emb_data['emb']


    train_val_data = ctr_data[ctr_data['type'] == 1]
    test_data = ctr_data[ctr_data['type'] == 2]

    train_size = int(len(train_val_data) * 0.9)
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:]

    item_id_field = "Movie ID"
    if data_name == "ml-1m":
        item_id_field = "Movie ID"
        train_dataset = MovieLensDataset(train_data, meta_data, is_vec=is_vec)
        val_dataset = MovieLensDataset(val_data, meta_data, is_vec=is_vec)
        test_dataset = MovieLensDataset(test_data, meta_data, is_vec=is_vec)

    elif data_name == "toys":
        item_id_field = "Item ID"
        train_dataset = ToysDataset(train_data, meta_data, is_vec=is_vec)
        val_dataset = ToysDataset(val_data, meta_data, is_vec=is_vec)
        test_dataset = ToysDataset(test_data, meta_data, is_vec=is_vec)
    
    elif data_name == "bookcross":
        item_id_field = "ISBN"
        train_dataset = BookcrossDataset(train_data, meta_data, is_vec=is_vec)
        val_dataset = BookcrossDataset(val_data, meta_data, is_vec=is_vec)
        test_dataset = BookcrossDataset(test_data, meta_data, is_vec=is_vec)
    
    else:
        raise ValueError("Acceptable dataset names are ['ml-1m', 'toys', 'bookcross']")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model_name == "dcn":
        model = DCNV2(num_fields=len(meta_data['field_names'])-1,
                    embed_size=32,
                    input_size=sum(meta_data['feature_count']),
                    embed_dropout_rate=0.1,
                    num_cross_layers=6,
                    num_hidden_layers=6,
                    hidden_size=256,
                    hidden_dropout_rate=0.1,
                    hidden_act='relu',
                    output_dim=1,
                    vec_dim=vec_dim,
                    fusion_type=fusion_type)

    criterion = torch.nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if scheduler_type == 'cosine':
        scheduler_params = {
            'T_max': 100,
            'eta_min': 1e-6
        }
    elif scheduler_type == 'linear':
        scheduler_params = {
            'start_factor': 0.1,
            'end_factor': 1.0,
            'total_iters': 100
        }
    else:
        scheduler_params = {}

    trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device=device, logger=logger, is_vec=is_vec, patience=2,
                      scheduler_type=scheduler_type, model_save_path=model_save_path, **scheduler_params)

    if early_stop:
        trainer.train_early_stop(epochs=max_epoch)
    else:
        trainer.train(epochs=max_epoch)
    trainer.test(test_loader=test_loader, model_path=model_save_path)

if __name__ == '__main__':
    main()
