from sklearn.model_selection import train_test_split
from sksurv.svm  import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["event"], y["time"], prediction)
    return result[0]

def one_split(Xtrain, ytrain, Xval, yval, model):

    model.fit(Xtrain, ytrain)

    predicted_risk_scores = model.predict(Xval)
    c_index = concordance_index_censored(yval['event'], yval['time'], predicted_risk_scores)[0]

    return c_index

def train_evaluate(X, y, groups, sgkf, model):
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y['event'], groups)):
        Xtrain = X.iloc[train_idx]
        ytrain = y[train_idx]
        Xval = X.iloc[val_idx]
        yval = y[val_idx]

        c_idx = one_split(Xtrain.values, ytrain, Xval.values, yval, model)

        fold_results = {
            'fold': fold_idx,
            'c_index': c_idx
        }

        results.append(fold_results)

    df = pd.DataFrame(results)

    # Add summary row
    summary = {
        'fold': 'mean',
        'c_index': df['c_index'].mean()
    }
    stds = {
        'fold': 'std',
        'c_index': df['c_index'].std()
    }

    df = pd.concat([df, pd.DataFrame([summary, stds])], ignore_index=True)

    return df