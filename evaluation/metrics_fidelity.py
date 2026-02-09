#%%
import numpy as np
from scipy.stats import entropy

from modules import utils
# %%
"""
Reference:
[1] https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/utils/metrics.py
[2] Synthcity: facilitating innovative use cases of synthetic data in different data modalities
- https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_statistical.py
"""
#%%
def SMAPE(train_dataset, imputed):
    """continuous"""
    C = train_dataset.num_continuous_features
    original = train_dataset.raw_data.values[:, :C]
    original = original[train_dataset.mask[:, :C] == 1]
    
    imputation = imputed.values[:, :C]
    imputation = imputation[train_dataset.mask[:, :C] == 1]
    
    smape = np.abs(original - imputation)
    smape /= (np.abs(original) + np.abs(imputation)) + 1e-6 # numerical stability
    smape = smape.mean()
    
    """categorical"""
    original = train_dataset.raw_data.values[:, C:]
    original = original[train_dataset.mask[:, C:] == 1]
    
    imputation = imputed.values[:, C:]
    imputation = imputation[train_dataset.mask[:, C:] == 1]
    
    error = 1. - (original == imputation).mean()
    
    return smape, error
#%%
def KLDivergence(train_dataset, imputed):
    """
    Marginal statistical fidelity: KL-Divergence
    : lower is better
    """
    
    train = train_dataset.raw_data
    num_bins = 10
    
    # get distribution of continuous variables
    cont_freqs = utils.get_frequency(
        train[train_dataset.continuous_features], 
        imputed[train_dataset.continuous_features], 
        n_histogram_bins=num_bins
    )
    
    result = [] 
    for col in train.columns:
        if col in train_dataset.continuous_features:
            gt, syn = cont_freqs[col]
            kl_div = entropy(syn, gt)
        else:
            pmf_p = train[col].value_counts(normalize=True)
            pmf_q = imputed[col].value_counts(normalize=True)
            
            # Ensure that both PMFs cover the same set of categories
            all_categories = pmf_p.index.union(pmf_q.index)
            pmf_p = pmf_p.reindex(all_categories, fill_value=0)
            pmf_q = pmf_q.reindex(all_categories, fill_value=0)
            
            # Avoid division by zero and log(0) by filtering out zero probabilities
            non_zero_mask = (pmf_p > 0) & (pmf_q > 0)
            
            kl_div = np.sum(pmf_q[non_zero_mask] * np.log(pmf_q[non_zero_mask] / pmf_p[non_zero_mask]))
        result.append(kl_div)
    return np.mean(result)