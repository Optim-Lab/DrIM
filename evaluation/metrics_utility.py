"""
Reference:
[1] Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark
- https://github.com/HLasse/data-centric-synthetic-data
"""
#%%
import numpy as np

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr

#%%
def classification(train_dataset, test_dataset, imputed):
    continuous = train_dataset.continuous_features
    target = train_dataset.ClfTarget
    
    train_ = train_dataset.raw_data.copy()
    test_ = test_dataset.raw_data.copy()
    imputed_ = imputed.copy()
    
    mean = train_[continuous].mean()
    std = train_[continuous].std()
    
    std.replace(0, 1, inplace=True) # except for std=0

    train_[continuous] -= mean
    train_[continuous] /= std
    test_[continuous] -= mean
    test_[continuous] /= std
    imputed_[continuous] -= mean
    imputed_[continuous] /= std

    assert train_.isna().sum().sum() == 0
    assert test_.isna().sum().sum() == 0
    
    covariates = [x for x in train_.columns if x not in [target]]
    
    """Baseline"""
    performance = []
    print(f"\n(Baseline) Classification: F1...")
    for name, clf in [
        ('logit', LogisticRegression(random_state=0, n_jobs=-1, max_iter=1000)),
        ('GaussNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('tree', DecisionTreeClassifier(random_state=0)),
        ('RF', RandomForestClassifier(random_state=0)),
    ]:  
        clf.fit(train_[covariates], train_[target])
        pred = clf.predict(test_[covariates])
        f1 = f1_score(test_[target], pred, average='micro')
        if name == "RF":
            feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
        print(f"[{name}] F1: {f1:.3f}")
        performance.append((name, f1))

    base_performance = performance
    base_cls = np.mean([x[1] for x in performance])
    base_feature = feature
    
    """Imputed"""
    if imputed_[target].sum() == 0:
        return (
            base_cls, 0., 0., 0.
        )
    else:
        performance = []
        print(f"\n(Imputed) Classification: F1...")
        for name, clf in [
            ('logit', LogisticRegression(random_state=0, n_jobs=-1, max_iter=1000)),
            ('GaussNB', GaussianNB()),
            ('KNN', KNeighborsClassifier(n_jobs=-1)),
            ('tree', DecisionTreeClassifier(random_state=0)),
            ('RF', RandomForestClassifier(random_state=0)),
        ]:
            clf.fit(imputed_[covariates], imputed_[target])
            pred = clf.predict(test_[covariates])
            f1 = f1_score(test_[target], pred, average='micro')
            if name == "RF":
                feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
            print(f"[{name}] F1: {f1:.3f}")
            performance.append((name, f1))
                
        imputed_cls = np.mean([x[1] for x in performance])
        model_selection = spearmanr(
            np.array([x[1] for x in base_performance]),
            np.array([x[1] for x in performance])).statistic
        feature_selection = spearmanr(
            np.array([x[1] for x in base_feature]),
            np.array([x[1] for x in feature])).statistic
        
        return (
            base_cls, imputed_cls, model_selection, feature_selection
        )
#%%