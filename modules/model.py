#%%
import copy
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import cdist
from collections import namedtuple

from sklearn.metrics.pairwise import cosine_similarity
from transformers import (BertTokenizer, 
                          BertModel, 
                          GPT2Tokenizer, 
                          GPT2Model, 
                          LlamaTokenizerFast,
                          LlamaModel,
                          LlamaConfig,
                          GPTNeoModel,
                          AutoTokenizer,
                          RobertaTokenizer,
                          RobertaModel,
                          )

ModelInfo = namedtuple('ModelInfo', ['embedding_dim'])
#%%
class Imputer:
    def __init__(self, config):
        self.config = config
        self.tokenizer, self.language_model = self._initialize_language_model()

    def _initialize_language_model(self):
        """Initialize the language model"""

        if self.config["language_model"] == "bert-base":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            language_model = BertModel.from_pretrained("bert-base-uncased")
            self.ModelInfo = ModelInfo(768)
            
        elif self.config["language_model"] == "bert-large":
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            language_model = BertModel.from_pretrained("bert-large-uncased")
            self.ModelInfo = ModelInfo(1024)
            
        elif self.config["language_model"] == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            language_model = GPT2Model.from_pretrained("gpt2-xl")
            language_model.resize_token_embeddings(len(tokenizer))
            self.ModelInfo = ModelInfo(1600)
            
        elif self.config["language_model"] == "llama":
            tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            configuration = LlamaConfig()
            language_model = LlamaModel(configuration)
            language_model.resize_token_embeddings(len(tokenizer))
            self.ModelInfo = ModelInfo(4096)
            
        elif self.config["language_model"] == "gpt-neo":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            language_model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
            language_model.resize_token_embeddings(len(tokenizer))
            self.ModelInfo = ModelInfo(2048)
            
        elif self.config["language_model"] == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
            language_model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
            self.ModelInfo = ModelInfo(768)
            
        else:
            raise ValueError("Unsupported model type")
        
        return tokenizer, language_model
    #%%
    def imputer(self, train_dataset, embed_data):
        """
        Imputation
        
        Args
        ----------
            train_dataset : Data module with missing data.
            embed_data : embed missing data via BERT (language model).
        
        Returns
        ----------
            imputed : Data with imputed
        """
        embed_data = embed_data.astype(np.float32) # reducing the memory
        n = train_dataset.EncodedInfo.num_samples
        imputed = copy.deepcopy(train_dataset.imputed_data)
        
        metric = self.config["metric"]
        
        if metric == "cosine":
            similarity_matrix = cosine_similarity(embed_data)
            assert similarity_matrix.shape == (n, n)
            
            sorted_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:] # except for itself
            assert sorted_indices.shape == (n, n-1)
        else:
            if metric == "manhattan":
                dist_matrix = cdist(embed_data, embed_data, metric='cityblock')
            elif metric == "chebyshev":
                dist_matrix = cdist(embed_data, embed_data, metric='chebyshev')
            elif metric == "correlation":
                # correlation distance: 1 - correlation coefficient
                dist_matrix = cdist(embed_data, embed_data, metric='correlation')
            elif metric == "mahalanobis":
                cov_matrix = np.cov(embed_data, rowvar=False)
                
                eps = 1e-6  # cov_matrix must be positive definite
                cov_matrix_reg = cov_matrix + eps * np.eye(cov_matrix.shape[0])
                VI_reg = np.linalg.inv(cov_matrix_reg)# inverse matrix of covariance

                L = np.linalg.cholesky(VI_reg) # Cholesky decomposition
                X = embed_data.dot(L)
                X_norm_squared = np.sum(X**2, axis=1)
                dist_matrix = np.sqrt(X_norm_squared[:, None] + X_norm_squared[None, :] - 2 * np.dot(X, X.T))
            else:
                raise ValueError("Unsupported metric: " + metric)
            
            np.fill_diagonal(dist_matrix, np.inf)
            sorted_indices = np.argsort(dist_matrix, axis=1)
                
        for idx in tqdm(range(n), desc="Imputation..."):
            obs = train_dataset.imputed_data.drop(
                index=idx, axis=0, inplace=False
            )
            sorted_obs = obs.reindex(sorted_indices[idx])

            imputed_features = train_dataset.imputed_data.iloc[idx][
                train_dataset.imputed_data.iloc[idx].isna()==True
            ].index.to_list()
        
            for feature in imputed_features:
                candidate_neighbors = sorted_obs[feature].dropna() # eliminating missing value
                
                # continuous
                if feature in train_dataset.continuous_features:
                    result = candidate_neighbors.head(self.config["K"]).mean() # top-k neighbors set

                # categorical
                else:
                    result = candidate_neighbors.head(self.config["K"]).mode()
                    if not result.empty:
                        result = result[0] # selection first mode in two mode case
                
                # post processing
                if feature in train_dataset.integer_features:
                    result = result.astype(int)

                imputed.loc[idx, feature] = result

        return imputed
    #%%
