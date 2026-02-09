#%%
import copy
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

import wandb
# %%
class density_ratio_r(nn.Module):
    """r(x, u)"""
    def __init__(self, embedding_dim):
        super(density_ratio_r, self).__init__()
        self.a_layers = nn.Linear(embedding_dim, 1)
        self.b_layers = nn.Linear(embedding_dim, 1)

    def forward(self, h_x, h_u):
        """
        Args:
            h_x: Representation of x (Anchor) (batch_size, embedding_dim)
            h_u: Representation of u (Positive/Negative) (batch_size, embedding_dim)
        return: 
            r(x, u) value
        """
        
        psi_ = torch.matmul(h_x, h_u.T)
        psi = torch.diag(psi_).view(-1, 1)
        
        # Calculate a(h_x) and b(h_u)
        a_value = self.a_layers(h_x)
        b_value = self.b_layers(h_u)
            
        return psi + a_value + b_value
# %%
def train_function_LR(
    textual_data,
    tokenizer,
    language_model, 
    optimizer, 
    config, 
    ModelInfo,
    device):
    #%%
    density_ratio_model = density_ratio_r(
        embedding_dim=ModelInfo.embedding_dim).to(device)

    """model specification"""
    if config["language_model"] in ["bert-base", "bert-large", "roberta"]:
        tune_layers = [
            language_model.encoder.layer[-i] for i in range(1, config["layers"] + 1)
        ]
        
    elif config["language_model"] in ["gpt2", "gpt-neo"]:
        tune_layers = [
            language_model.h[-i] for i in range(1, config["layers"] + 1)
        ]
        
    elif config["language_model"] in ["llama"]:
        tune_layers = [
            language_model.layers[-i] for i in range(1, config["layers"] + 1)
        ]
    elif config["language_model"] in ["opt"]:
        tune_layers = [
            language_model.decoder.layers[-i] for i in range(1, config["layers"] + 1)
        ]
        
    """the number of fine-tune layers"""
    for param in language_model.parameters():
            param.requires_grad = False
    
    for layer in tune_layers: 
        for param in layer.parameters():
            param.requires_grad = True
    #%%
    """train"""
    bceloss = nn.BCEWithLogitsLoss().to(device)
    n = len(textual_data) 

    for epoch in range(config["epochs"]):
        logs = {
            'bce_loss': [], 
        }        
        for i in tqdm(range(0, n, config["batch_size"]), desc="inner loop..."):
            bce_loss_ = []
            
            batch_texts = textual_data[i:min(i+config["batch_size"], n)]
            batch_size_ = len(batch_texts)           
            
            if config["encoding"] == "tab2text":
                """positive sample""" 
                positive_texts = []
                for j in range(batch_size_): # last batch is not equal to length of batch_texts
                    positive = batch_texts[j]
                    positive = parser(positive)
                    if config["language_model"] in ["bert-base", "bert-large", "roberta"]: 
                        positive = remask(positive) # [mask] token
                    else:
                        positive = remask_unknown(positive) # [unk] token
                        
                    positive = reformat(positive)
                    positive_texts.append(positive) 
                
                # Debugging: the positive sample is really valuable?
                if not debug_positive(positive_texts):
                    print("all positive text is masking...")
                    continue
                            
                if len(positive_texts) == 1: # empty negative sample
                    continue
                
                """negative sample"""
                negative_texts = permute_positive(positive_texts)
            
            elif config["encoding"] == "tab2json":
                
                """positive sample"""       
                positive_texts = []
                for j in range(batch_size_):
                    positive = batch_texts[j]
                    if config["language_model"] in ["bert-base", "bert-large", "roberta"]: 
                        positive = remask(positive)  # [MASK] token 
                    else:
                        positive = remask_unknown(positive)  # [UNK] token
                        
                    positive = str(positive)
                    positive_texts.append(positive) 
                
                # for debugging
                if not debug_positive(positive_texts):
                    print("all positive text is masking...")
                    continue
                            
                if len(positive_texts) == 1:  
                    continue
                
                """negative sample"""
                negative_texts = permute_positive(positive_texts)

                batch_texts = [str(item) for item in batch_texts] ### dicitionary to string type for tonkenizer
            
            elif config["encoding"] == "tab2num":
                """positive sample""" 
                positive_texts = []
                for j in range(batch_size_):# last batch is not equal to length of batch_texts
                    positive = batch_texts[j]
                    if config["language_model"] in ["bert-base", "bert-large", "roberta"]: 
                        positive = remask_numerical(positive) # [mask] token
                    else:
                        positive = remask_unknown_numerical(positive) # [unk] token
                        
                    positive_texts.append(positive) 
                
                # Debugging: the positive sample is really valuable?
                if not debug_positive(positive_texts):
                    print("all positive text is masking...")
                    continue
                            
                if len(positive_texts) == 1: # empty negative sample
                    continue
                
                """negative sample"""
                negative_texts = permute_positive(positive_texts)
                
            """tokenizing"""
            batch_ = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )

            positive_ = tokenizer(
                positive_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )

            negative_ = tokenizer(
                negative_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            """representation"""
            outputs = language_model(
                **{key: value.to(device) for key, value in batch_.items()}
            )
            if config["language_model"] in ["bert-base", "bert-large", "roberta"]:
                anchor = outputs.last_hidden_state[:, 0, :] # [batch, 768]
            else:
                anchor = outputs.last_hidden_state.mean(dim=1)
            positive_outputs = language_model(
                **{k: v.to(device) for k, v in positive_.items()}
            )
            if config["language_model"] in ["bert-base", "bert-large", "roberta"]:
                positive = positive_outputs.last_hidden_state[:, 0, :] # [batch, 768]
            else:
                positive = positive_outputs.last_hidden_state.mean(dim=1)
                
            negative_outputs = language_model(
                **{k: v.to(device) for k, v in negative_.items()}
            )
            if config["language_model"] in ["bert-base", "bert-large", "roberta"]:
                negative = negative_outputs.last_hidden_state[:, 0, :] # [batch]
            else:
                negative = negative_outputs.last_hidden_state.mean(dim=1)

            optimizer.zero_grad()
            
            """r(t, u)"""
            r_positive = density_ratio_model(anchor, positive) # [batch, 1]
            
            """r(t, u')"""
            r_negative = density_ratio_model(anchor, negative) # [batch, 1]
            
            logit = torch.cat([r_positive, r_negative], dim=0).view(-1) # [batch, 1(positive) + num_negatives]
            label = torch.cat(
                [torch.ones(batch_size_), torch.zeros(batch_size_)]).to(device)
            
            loss = bceloss(logit, label)
            loss.backward()
            optimizer.step()

            bce_loss_.append(('bce_loss', loss))    
            
            """accumulate losses"""
            for x, y in bce_loss_:
                logs[x] = logs.get(x) + [y.item()]         

        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)

        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    
    return 
# %%
def parser(textual):
    items = textual.split(', ')
    data = {}
    for item in items:
        key, value = item.split(' is ')
        try:
            data[key] = float(value)
        except ValueError:
            data[key] = value
    return data

def remask(data):
    """generate a positive sample BERT-based model"""
    cols = list(data.keys())
    p = len(cols)

    missing_count = list(data.values()).count('[MASK]')
    
    rate = (p - missing_count) / 2
    n_mask = sample_zero_truncated_poisson(rate)
    
    non_missing_cols = [
        key for key, value in data.items() if value != '[MASK]']
    
    n_mask = min(n_mask, len(non_missing_cols)) # case: non_missing cols < n_masks 
    
    remasked_cols = random.sample(non_missing_cols, n_mask)
    remasked = copy.deepcopy(data)

    for col in remasked_cols:
        # if remasked[col] == '[MASK]':
        #     continue  # skip existing [MASK] value 
        remasked[col] = '[MASK]'
        
    return remasked

def remask_unknown(data):
    """generate a positive sample GPT-based model"""
    cols = list(data.keys())
    p = len(cols)

    missing_count = list(data.values()).count('[UNK]')
    
    rate = (p - missing_count) / 2
    n_mask = sample_zero_truncated_poisson(rate)
    
    non_missing_cols = [
        key for key, value in data.items() if value != '[UNK]']
    
    n_mask = min(n_mask, len(non_missing_cols)) # case: non_missing cols < n_masks 
    
    remasked_cols = random.sample(non_missing_cols, n_mask)
    remasked = copy.deepcopy(data)

    for col in remasked_cols:
        # if remasked[col] == '[MASK]':
        #     continue  # skip existing [MASK] value 
        remasked[col] = '[UNK]'
        
    return remasked

def remask_numerical(data):
    p = data.count(',') + 1
    
    missing_count = data.count('[MASK]')
    
    rate = (p - missing_count) / 2
    n_mask = sample_zero_truncated_poisson(rate)
    
    n_mask = min(n_mask, p - missing_count) # case: non_missing cols < n_masks 
    values = [val.strip() for val in data.split(',')]
    non_mask_indices = [i for i, val in enumerate(values) if val != "[MASK]"]
    indices_to_mask = random.sample(non_mask_indices, min(3, len(non_mask_indices)))
    
    for i in indices_to_mask:
        values[i] = "[MASK]"

    remasked = ", ".join(values)
    
    return remasked

def remask_unknown_numerical(data):
    p = data.count(',') + 1
    
    missing_count = data.count('[UNK]')
    
    rate = (p - missing_count) / 2
    n_mask = sample_zero_truncated_poisson(rate)
    
    n_mask = min(n_mask, p - missing_count) # case: non_missing cols < n_masks 
    values = [val.strip() for val in data.split(',')]
    non_mask_indices = [i for i, val in enumerate(values) if val != "[UNK]"]
    indices_to_mask = random.sample(non_mask_indices, min(3, len(non_mask_indices)))
    
    for i in indices_to_mask:
        values[i] = "[UNK]"

    remasked = ", ".join(values)
    
    return remasked

def reformat(data):
    """Format dictionary back into string format"""
    return ', '.join(f"{key} is {value}" for key, value in data.items())

def debug_positive(positive_texts):
    # if all mask positive text
    # then no valuable negative text is employed
    first_positive = positive_texts[0]
    for element in positive_texts[1:]:
        if element != first_positive:
            return True
    return False

def permute_positive(positive_texts):
    """generate a negative sample"""
    while True:
        negative_texts = random.sample(positive_texts, len(positive_texts))
        if negative_texts != positive_texts:
            return negative_texts
    
def sample_zero_truncated_poisson(rate):
    u = np.random.uniform(np.exp(-rate), 1)
    t = -np.log(u)
    return 1 + np.random.poisson(rate - t)