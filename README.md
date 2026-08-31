# DrIM
This repository is the official implementation of 'DrIM: Context-Driven Nearest Neighbor Imputation using Language Representation' with PyTorch (PAKDD 2026).

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Overview
<img src="Figure.png" alt="image" width="700"/>

## Dataset

Download and add the datasets into `data` folder to reproduce our experimental results.

## Reproducibility

### Arguments
- `--dataset`: dataset options (`abalone`,  `anuran`, `banknote`, `breast`, `concrete`, `kings`,  `letter`, `loan`, `redwine`, `whitewine`)
- `--missing_type`: how to generate missing (`MCAR`, `MAR`, `MNARL`, `MNARQ`)
- `--missing_rate`: missingness rate (default: `0.3`)
- `--layers`: the number of layers fine-tuned in language model (default: `3`)
- `--language_model`: Language model (default: `bert-base`), options (`bert-base`, `bert-large`, `gpt2`, `llama`, `gpt-neo`, `roberta`)
- `--K`: the number of nearest neighbors (default: `5`)

### Imputation & Evaluation 

> RQ1. ***Overall performance.*** Does DrIM demonstrate state-of-the-art performance in missing data imputation? 
```
python main.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate>
```

> RQ2. ***Ablation study: Effect of contrastive learning.*** To what extent does contrastive learning contribute to the imputation performance of DrIM? 
- w/o CL
```
python main.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> --layers 0
```
- DrIM
```
python main.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> --layers 3
```

> RQ3. ***Sensitivity analysis: Missingness scenarios.*** How robust is DrIM's performance under varying missingness rates and patterns? 
```
python main.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate>
```


> RQ4. ***Ablation study: Language models.*** How does DrIM perform when combined with different language models? 
```
python main.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> --language_model <language_model>
```

## Directory and codes

```
.
+-- data
+-- assets 
+-- datasets
|       +-- preprocess.py
|       +-- raw_data.py
+-- evaluation
|       +-- evaluation.py
|       +-- metrics_impute.py
|       +-- metrics_MLu.py
+-- modules 
|       +-- embedding.py
|       +-- missing.py
|       +-- model.py
|       +-- textual_encoding.py
|       +-- train.py
|       +-- utils.py
+-- main.py
+-- supp.pdf
+-- Figure.png
+-- README.md
```

## Citation
```bibtex
@inproceedings{10.1007/978-981-92-1462-4_35,
author = {Lim, Jaesung and An, Seunghwan and Jeon, Jong-June},
title = {DrIM: Context-Driven Nearest Neighbor Imputation Using Language Representation},
year = {2026},
isbn = {978-981-92-1461-7},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-981-92-1462-4_35},
doi = {10.1007/978-981-92-1462-4_35},
abstract = {Missing data poses significant challenges for machine learning and deep learning algorithms, which require complete datasets for training. In this paper, we aim to enhance post-imputation performance, measured by imputation utility. We introduce a k-nearest neighbors-based imputation method, DrIM, designed for heterogeneous (mixed-type) tabular datasets. DrIM leverages the representation learning capabilities of language models by transforming the tabular dataset into a text format and replacing the missing entries with [MASK] (or [UNK]) tokens. DrIM incorporates a contrastive learning framework and refines the representations. Moreover, our proposed method is theoretically justified by showing that contrastive learning induces a metric of representation space via density ratio estimation, thereby supporting its use for missing data imputation. To validate our proposed model, we evaluate its performance on missing data imputation across 10 real-world tabular datasets, demonstrating its ability to produce complete datasets with high imputation utility under various missing data scenarios.},
booktitle = {Advances in Knowledge Discovery and Data Mining: 30th Pacific-Asia Conference on Knowledge Discovery and Data Mining, PAKDD 2026, Hong Kong, China, June 9–12, 2026, Proceedings, Part II},
pages = {444–457},
numpages = {14},
keywords = {Missing data imputation, Nearest neighbor, Language representation, Mutual information, Contrastive learning},
location = {Hong Kong, China}
}
```

## Acknowledgement
This repository was developed with support from the 서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단 - http://dsplus.uos.ac.kr/
