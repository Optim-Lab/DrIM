#%%
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from collections import namedtuple

from sklearn.model_selection import train_test_split

from modules.textual_encoding import tab2text, tab2json, tab2num
from modules.missing import generate_mask
from datasets.raw_data import load_raw_data
#%%
EncodedInfo = namedtuple('EncodedInfo', ['num_samples'])
#%%
class TextualDataset(Dataset):
    def __init__(
        self, 
        config,
        train=True):

        self.config = config
        self.train = train
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config)

        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = continuous_features + categorical_features
        self.num_continuous_features = len(continuous_features)

        """categorical"""
        self.category_maps = {}
        for feature in categorical_features:
            data[feature] = data[feature].astype('category') # convert to category type
            self.category_maps[feature] = dict(
                enumerate(data[feature].cat.categories)) # save the category map
            data[feature] = data[feature].cat.codes 

        self.num_categories = data[categorical_features].nunique(axis=0).to_list()

        """train test split"""
        data = data[self.features]
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        data = train_data if train else test_data
        data = data.reset_index(drop=True)
        self.raw_data = copy.deepcopy(data) # copy for non-masking & evaluation (ground truth)

        """generating missing value"""
        assert config["missing_type"] != "None"
        
        mask = generate_mask(
            X_true=torch.from_numpy(data.values).float(), 
            config=config)
        data.mask(mask.astype(bool), np.nan, inplace=True)
        self.mask = mask    
        self.imputed_data = copy.deepcopy(data) # pd.DataFrame type for imputation

        for feature in categorical_features:
            data[feature] = data[feature].map(self.category_maps[feature])

        """[MASK] or [UNK] token"""
        if config["language_model"] in ["bert-base", "bert-large", "roberta"]:
            for feature in self.features:
                data[feature] = data[feature].apply(
                    lambda x: '[MASK]' if pd.isna(x) else x) 
        else:
            for feature in self.features:
                data[feature] = data[feature].apply(
                    lambda x: '[UNK]' if pd.isna(x) else x) 

        """textual encoding"""
        if config["encoding"] == "tab2text":             
            textual_data = []
            for idx in tqdm(range(len(data)), desc="textual encoding..."):
                text = tab2text(
                    missing=data, 
                    idx=idx
                )
                textual_data.append(text)
            
            self.data = textual_data
            
        elif config["encoding"] == "tab2json":
            json_data = []
            for idx in tqdm(range(len(data)), desc="JSON formatting..."):
                text = tab2json(
                    missing=data, 
                    idx=idx
                )
                json_data.append(text)
            
            self.data = json_data
            
        elif config["encoding"] == "tab2num":
            num_data = []
            for idx in tqdm(range(len(data)), desc="numerical encoding..."):
                text = tab2num(
                    missing=data, 
                    idx=idx
                )
                num_data.append(text)
                
            self.data = num_data
        self.EncodedInfo = EncodedInfo(len(self.imputed_data))
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self):
        return self.data
#%%
