import pandas as pd
import numpy as np
import argparse
import qlib
import pandas as pd
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
import os

#make argparse
train_start, train_end = "2005-01-01", "2014-12-31"

valid_start, valid_end = "2015-01-01", "2016-12-31"
 
test_start, test_end = "2017-01-01", "2020-09-23"
# start_time = train_start
# end_time = test_end
# fit_start_time = train_start
# fit_end_time = train_end

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./data', help='directory to save model')
parser.add_argument('--start_time', type=str, default='2005-01-01', help='start time')
parser.add_argument('--end_time', type=str, default='2020-09-23', help='end time')
parser.add_argument('--fit_end_time', type=str, default='2014-12-31', help='fit end time')

parser.add_argument('--val_start_time', type=str, default='2015-01-01', help='val start time')
parser.add_argument('--val_end_time', type=str, default='2016-12-31', help='val end time')

parser.add_argument('--seq_len', type=int, default=20, help='sequence length')

parser.add_argument('--normalize', action='store_true', help='whether to normalize')
parser.add_argument('--select_feature', action='store_true', help='whether to select feature')
parser.add_argument('--use_qlib', action='store_true', help='whether to use qlib data')
args = parser.parse_args()


def main(args):
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    # provider_uri = "/home/alvinma/Desktop/qlibtutor/qlib_data/cn_data" 
    qlib.init(provider_uri=provider_uri, region=REG_CN)
 
    market = "csi300"
    benchmark = "SH000300"
    if args.select_feature:
        data_handler_config = {
        "start_time": f"{args.start_time}",
        "end_time": f"{args.end_time}",
        "fit_start_time": f"{args.start_time}",
        "fit_end_time": f"{args.fit_end_time}",
        "instruments": "csi300",
        "infer_processors": [{"class": 'FilterCol', 'kwargs' : {'col_list': ['RESI5','WVMA5','RSQR5','KLEN','RSQR10','CORR5','CORD5','CORR10','ROC60','RESI10','VSTD5','RSQR60','CORR60','WVMA60','STD5','RSQR20','CORD60','CORD10','CORR20','KLOW'], 'fields_group': 'feature'}},
                                {"class": 'RobustZScoreNorm', 'kwargs' : {'clip_outlier': True, 'fields_group': 'feature'}},
                                {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
        "learn_processors": [{"class": 'DropnaLabel'},
                                {"class": 'CSRankNorm', 'kwargs' : {'fields_group': 'label'}}],
        }
    else:
        data_handler_config = {
        "start_time": f"{args.start_time}",
        "end_time": f"{args.end_time}",
        "fit_start_time": f"{args.start_time}",
        "fit_end_time": f"{args.fit_end_time}",
        "instruments": "csi300",
        "infer_processors": [{"class": 'RobustZScoreNorm', 'kwargs' : {'clip_outlier': True, 'fields_group': 'feature'}},
                                {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
        "learn_processors": [{"class": 'DropnaLabel'},
                                {"class": 'CSRankNorm', 'kwargs' : {'fields_group': 'label'}}],
        }
    
    dataset = Alpha158(**data_handler_config)
    
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
    
    if not args.use_qlib:
        
        if args.normalize:
            dataframe = dataset.fetch(col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
        else:
            dataframe = dataset.fetch(col_set=["feature","label"])
        

        dataframe.columns = dataframe.columns.droplevel(0)
        
        # ! 자동 조정이 생각보다 힘든 부분. 실제로 데이터를 보고 조정해야함.
        train = dataframe.loc[pd.IndexSlice[ : f'{args.fit_end_time}', :], :]
        
        val_start_time = pd.to_datetime(args.val_start_time) # - pd.DateOffset(days=30)
        valid = dataframe.loc[pd.IndexSlice[val_start_time: f'{args.val_end_time}', :], :]
        
        test_start_time = pd.to_datetime(args.val_end_time) # - pd.DateOffset(days=30)
        test = dataframe.loc[pd.IndexSlice[test_start_time:, :], :]
        print(f"shape of dataframe: {dataframe.shape} time: {dataframe.index[0]} ~ {dataframe.index[-1]} memory: {dataframe.memory_usage().sum() / 1024 ** 2} MB")
        print(f"shape of train: {train.shape} time: {train.index[0]} ~ {train.index[-1]} memory: {train.memory_usage().sum() / 1024 ** 2} MB")
        print(f"shape of valid: {valid.shape} time: {valid.index[0]} ~ {valid.index[-1]} memory: {valid.memory_usage().sum() / 1024 ** 2} MB")
        print(f"shape of test: {test.shape} time: {test.index[0]} ~ {test.index[-1]} memory: {test.memory_usage().sum() / 1024 ** 2} MB")


    else:
        from qlib.data.dataset import DatasetH, TSDatasetH
        # r_data_h = TSDatasetH(handler=dataset, segments={"train": ("2010-01-01", "2017-12-31"), \
        #                                                 "valid": ("2018-01-01", "2018-12-31"), \
        #                                                 "test": ("2019-01-01", "2020-10-01")}, step_len=args.seq_len)
        r_data_h = TSDatasetH(handler=dataset, segments={"train": (train_start, train_end), \
                                                        "valid": (valid_start, valid_end), \
                                                        "test": (test_start, test_end)}, step_len=args.seq_len)
        
        
        
        train = r_data_h.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        valid = r_data_h.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        test = r_data_h.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        print(f"type of train: {type(train)} ")

        r_data_h.to_pickle(f'{args.save_dir}/csi300_QLIB_{args.use_qlib}_NORM_{args.normalize}_CHAR_{args.select_feature}_LEN_{args.seq_len}.pkl')
        exit(0)
    
    train.to_pickle(f'{args.save_dir}/train_csi300_QLIB_{args.use_qlib}_NORM_{args.normalize}_CHAR_{args.select_feature}_LEN_{args.seq_len}.pkl')
    valid.to_pickle(f'{args.save_dir}/valid_csi300_QLIB_{args.use_qlib}_NORM_{args.normalize}_CHAR_{args.select_feature}_LEN_{args.seq_len}.pkl')
    test.to_pickle(f'{args.save_dir}/test_csi300_QLIB_{args.use_qlib}_NORM_{args.normalize}_CHAR_{args.select_feature}_LEN_{args.seq_len}.pkl')
    
    print(f"shape of train: {train.shape} time: {train.index[0]} ~ {train.index[-1]} memory: {train.memory_usage().sum() / 1024 ** 2} MB")
    print(f"shape of valid: {valid.shape} time: {valid.index[0]} ~ {valid.index[-1]} memory: {valid.memory_usage().sum() / 1024 ** 2} MB")
    print(f"shape of test: {test.shape} time: {test.index[0]} ~ {test.index[-1]} memory: {test.memory_usage().sum() / 1024 ** 2} MB")

    
if __name__ == "__main__":
    main(args)