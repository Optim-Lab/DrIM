#%%
from typing import List
from tqdm import tqdm

import numpy as np

import torch
#%%
import warnings
warnings.filterwarnings(
    'ignore', category=FutureWarning, module='huggingface_hub.file_download'
)
#%%
def get_embeddings(
    config,
    textual_data: List[str],
    tokenizer,
    language_model, 
    batch_size,
    device) -> np.ndarray:
    """
    embedding vector mini-batchs ver.

    Args
    ----------
        textual_data : Data with text form.
        batch_size : embeded batch size.

    Returns
    ----------
        X_embeded
    """
    
    embeddings = []
    
    for i in tqdm(range(0, len(textual_data), batch_size), desc="embedding..."):
        batch_texts = textual_data[i:i+batch_size]
        
        if not batch_texts:  # ignore the emtpy list
            continue
        
        encoded_inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = language_model(input_ids, attention_mask=attention_mask)
        if config["language_model"] in ["bert-base", "bert-large", "roberta"]:
            batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        else:
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)
# %%