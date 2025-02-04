import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
import argparse
from Layers import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
# from stockdata import StockDataset
from dataset import StockDataset
from train_model import train, validate, test
from utils import set_seed, DataArgument
# import wandb
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.utils import flatten_dict
from dataclasses import dataclass
parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')

parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--num_latent', type=int, default=158, help='number of latent variables')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--num_factor', type=int, default=60, help='number of factors')
parser.add_argument('--hidden_size', type=int, default=60, help='hidden size')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--run_name', type=str, help='name of the run')
parser.add_argument('--save_dir', type=str, default='./best_models', help='directory to save model')
parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
parser.add_argument('--normalize', action='store_true', help='whether to normalize the data')
#--use_qlib
parser.add_argument('--use_qlib', action='store_true', help='whether to use qlib')
args = parser.parse_args()

data_args = DataArgument(use_qlib=False, normalize=True, select_feature=False)

assert args.seq_len == data_args.seq_len, "seq_len in args and data_args must be the same"
assert args.normalize == data_args.normalize, "normalize in args and data_args must be the same"
        
if args.normalize:
    print("*************** Use normalized data ***************")
    print("select_feature:", data_args.select_feature)
    train_df = pd.read_pickle(f"{data_args.save_dir}/train_csi300_QLIB_{data_args.use_qlib}_NORM_{data_args.normalize}_CHAR_{data_args.select_feature}_LEN_{args.seq_len}.pkl")
    valid_df = pd.read_pickle(f"{data_args.save_dir}/valid_csi300_QLIB_{data_args.use_qlib}_NORM_{data_args.normalize}_CHAR_{data_args.select_feature}_LEN_{args.seq_len}.pkl")
    test_df = pd.read_pickle(f"{data_args.save_dir}/test_csi300_QLIB_{data_args.use_qlib}_NORM_{data_args.normalize}_CHAR_{data_args.select_feature}_LEN_{args.seq_len}.pkl")
    
else:
    print("Use raw data")
    train_df = pd.read_pickle('./data/train_csi300.pkl')
    valid_df = pd.read_pickle('./data/valid_csi300.pkl')
    test_df = pd.read_pickle('./data/test_csi300.pkl')
    
if args.wandb:
    wandb.init(project="FactorVAE", config=args, name=f"{args.run_name}")
    wandb.config.update(args)
    # wandb.log({"train_df": train_df, "valid_df": valid_df, "test_df": test_df})


def main(args, data_args):
    
    set_seed(args.seed)
    # make directory to save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # create model
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_latent, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.batch_size, args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    
    # create dataloaders
    # Assuming you want to create a mini-batch of size 300
    train_ds = StockDataset(train_df, args.batch_size, args.seq_len)
    valid_ds = StockDataset(valid_df, args.batch_size, args.seq_len)
    #test_ds = StockDataset(test_df, args.batch_size, args.seq_len)
    
    if args.use_qlib == True:        
        import qlib
        from qlib.data.dataset import DatasetH, TSDatasetH
        provider_uri = "/home/alvinma/Desktop/qlibtutor/qlib_data/cn_data" 
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        train_start, train_end = "2005-01-01", "2014-12-31"

        valid_start, valid_end = "2015-01-01", "2016-12-31"
        
        test_start, test_end = "2017-01-01", "2020-09-23"
        market = "csi300"
        benchmark = "SH000300"
        data_handler_config = {
            "start_time": f"{train_start}",
            "end_time": f"{test_end}",
            "fit_start_time": f"{train_start}",
            "fit_end_time": f"{train_end}",
            "instruments": "csi300",
            "infer_processors": [{"class": 'RobustZScoreNorm', 'kwargs' : {'clip_outlier': True, 'fields_group': 'feature'}},
                                    {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
            "learn_processors": [{"class": 'DropnaLabel'},
                                    {"class": 'CSRankNorm', 'kwargs' : {'fields_group': 'label'}}],
            }
    #         infer_processors:
    #     - class: RobustZScoreNorm
    #       kwargs:
    #           fields_group: feature
    #           clip_outlier: true
    #     - class: Fillna
    #       kwargs:
    #           fields_group: feature
    # learn_processors:
    #     - class: DropnaLabel
    #     - class: CSRankNorm
    #       kwargs:
    #           fields_group: label
        dataset = Alpha158(**data_handler_config)

        r_data_h = TSDatasetH(handler=dataset, segments={"train": (train_start, train_end), \
                                                            "valid": (valid_start, valid_end), \
                                                            "test": (test_start, test_end)}, step_len=args.seq_len)
            
        train_ds = r_data_h.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        valid_ds = r_data_h.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        if train_ds.empty or valid_ds.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        train_ds.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        valid_ds.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        wl_train = np.ones(len(train_ds))
        wl_valid = np.ones(len(valid_ds))
        # test = r_data_h.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        
    train_dataloader = DataLoader(train_ds, batch_size=300, shuffle=False, num_workers=4)
    valid_dataloader = DataLoader(valid_ds, batch_size=300, shuffle=False, num_workers=4)
    #test_dataloader = DataLoader(test_ds, batch_size=300, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
        
    factorVAE.to(device)
    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs)
    
    # Start Trainig
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(factorVAE, train_dataloader, optimizer, args)
        val_loss = validate(factorVAE, valid_dataloader, args)
        test_loss = np.NaN #test(factorVAE, test_dataloader, args)
        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") #Test Loss: {test_loss:.4f},
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #? save model in save_dir
            
            #? torch.save
            save_root = os.path.join(args.save_dir, f'{args.run_name}_{args.num_factor}_norm_{data_args.normalize}_char_{data_args.select_feature}.pt')
            torch.save(factorVAE.state_dict(), save_root)
            
        if args.wandb:
            wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss}) #, "Test Loss": test_loss})
    
    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()
    
if __name__ == '__main__':
    main(args, data_args)
