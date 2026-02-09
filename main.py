#%%
import os
import argparse
import importlib

import torch
import torch.optim as optim
import numpy as np

from modules import utils
from modules.utils import set_random_seed
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "DrIM" # put your WANDB project name
entity = "xxx" # put your WANDB username

run = wandb.init(
    project=project,
    entity=entity, 
    tags=["imputation"], # put tags of this python project
)
#%%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=0, 
                        help='model version number')
    
    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine,
                        yeast, nomao
                        """)
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")
    parser.add_argument('--shuffle', type=str2bool, default=False, 
                        help='feature shuffle in textual encoding')
    
    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="missingness mechanism options: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missingness rate") 
    
    parser.add_argument('--epochs', default=5, type=int,
                        help='the number of epochs to fine-tuning the language model.')
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="learning rate")

    parser.add_argument("--language_model", default="bert-base", type=str,
                        help="""
                        model options: 
                        bert-base, bert-large, gpt2, llama, gpt-neo, roberta
                        """)
    parser.add_argument("--layers", default=3, type=int,
                        help="# layers fine-tuned in BERT (0-12)") 
    
    parser.add_argument("--K", default=5, type=int,
                        help="the number of Nearest negihbors") 
    
    parser.add_argument("--metric", default="cosine", type=str,
                        help="""Distance metric for imputation. 
                        Options: 'cosine', 'manhattan', 'chebyshev', 'correlation', 'mahalanobis'
                        """)
    
    parser.add_argument("--encoding", default="tab2text", type=str,
                        help="""encoding strategy
                        Options: 
                        'tab2text (textual encoding), 
                        tab2json (JSON formatting), 
                        tab2num (Directly numerical embedding)'
                        """)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    
    assert config["missing_type"] != None
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    TextualDataset = dataset_module.TextualDataset

    train_dataset = TextualDataset(
        config, 
        train=True
    )

    test_dataset = TextualDataset(
        config,
        train=False
    )
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.Imputer(config)
    model.language_model.to(device)
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(model.language_model)
    print(f"Number of Parameters: {num_params/1000000000:.2f}B")
    wandb.log({"Number of Parameters": num_params/1000000000})
    #%%
    """embedding load"""
    base_name = f"{config['language_model']}_{config['layers']}_{config['missing_type']}_{config['missing_rate']}_{config['dataset']}"
    embed_data_dir = f"./assets/embed_data/{config['language_model']}/{config['dataset']}"
    embed_data_name = f"embed_data_{base_name}_{config['seed']}"
    embed_data_path = f"{embed_data_dir}/{embed_data_name}.npy"
    
    artifact = wandb.Artifact(
        "_".join(embed_data_name.split("_")[:-1]), 
        type='dataset', 
        metadata=config
    )
    #%%
    if os.path.exists(embed_data_path):
        embed_data = np.load(embed_data_path)
        artifact.add_file(embed_data_path)
        print("Loaded embeddings from file.")
    else:
        """fine-tuning"""
        if config["layers"] != 0:
            model.language_model.train()
            optimizer = optim.AdamW(model.language_model.parameters(), lr=config["lr"])

            train_module = importlib.import_module('modules.train')
            importlib.reload(train_module)
            train_module.train_function_LR(
                textual_data=train_dataset.data, 
                tokenizer=model.tokenizer,
                language_model=model.language_model, 
                optimizer=optimizer, 
                config=config, 
                ModelInfo=model.ModelInfo,
                device=device
            )
            artifact.add_file('./modules/train.py')

        """embedding"""
        model.language_model.eval()
        embedding_module = importlib.import_module('modules.embedding')
        importlib.reload(embedding_module)
        get_embeddings = embedding_module.get_embeddings

        embed_data = get_embeddings(
            config=config,
            textual_data=train_dataset.data, 
            tokenizer=model.tokenizer,
            language_model=model.language_model,
            batch_size=config["batch_size"],
            device=device
        )

    """embedding save"""
    if not os.path.exists(embed_data_dir):
        os.makedirs(embed_data_dir)
    np.save(f"./{embed_data_dir}/{embed_data_name}.npy", embed_data)
    artifact.add_file(f"./{embed_data_dir}/{embed_data_name}.npy")
    print("Saved and uploaded new embeddings.")
    
    artifact.add_file('./modules/model.py')
    artifact.add_file('./main.py')
    wandb.log_artifact(artifact)
    #%%
    """imputation"""
    imputed = model.imputer(train_dataset, embed_data)

    assert imputed.isna().sum().sum() == 0 
    #%%
    """evaluation"""    
    evaluate_module = importlib.import_module('evaluation.evaluation')
    importlib.reload(evaluate_module)
    evaluate = evaluate_module.evaluate

    results = evaluate(imputed, train_dataset, test_dataset, config, device)
    for x, y in results._asdict().items():
        print(f"{x}: {y:.4f}")
        wandb.log({f"{x}": y})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == "__main__":
    main()
# %%
