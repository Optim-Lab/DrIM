# %%
from collections import namedtuple
from evaluation import metrics_fidelity, metrics_utility

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

Metrics = namedtuple(
    "Metrics",
    [   
        "smape",
        "error",
        "asmape",
        "KL",
        "base_cls", 
        "imputed_cls",
        "model_selection", 
        "feature_selection",
    ]
)
#%%
def evaluate(imputed, train_dataset, test_dataset, config, device):
    
    print("\n1. Imputation Fidelity: ASMAPE...")
    smape, error = metrics_fidelity.SMAPE(train_dataset, imputed)
    asmape = smape + error

    print("\n2. Imputation Fidelity: KL-Divergence...")
    KL = metrics_fidelity.KLDivergence(train_dataset, imputed)
    
    print("\n3. Imputation Utility: Classification...")
    base_cls, imputed_cls, model_selection, feature_selection = metrics_utility.classification(train_dataset, test_dataset, imputed)

    return Metrics(
        smape, error, asmape, KL,
        base_cls, imputed_cls, model_selection, feature_selection,
    )