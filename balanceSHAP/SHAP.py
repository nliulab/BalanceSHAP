import shap
import torch
import numpy as np
# import pandas as pd
from tqdm import tqdm

def shap_values(model, x_background, x_explanation, explainer="DeepSHAP"):
    '''
    Compute SHAP values via deep explainer with background and explanation data (with targeted minority-majority ratio)
    
    ---
    Params:
    model: trained deep learning model
    x_background: tensor, predictors in background data
    x_explanantion: tensor, predictors in explanation data
    explainer: str, options: "DeepSHAP" (default), "GradientSHAP"
    '''
    if explainer == "DeepSHAP":
        e = shap.DeepExplainer(model, x_background)
    # print(e.expected_value)
    
        shap_list = []
        X = x_explanation
        batch_size = 1
        idxs = list(map(int, np.linspace(0, X.shape[0] - 1, int(X.shape[0]/batch_size))))
        
        for i1, i2 in zip(tqdm(idxs[:-1], desc="SHAP computation:"), idxs[1:]):
            shaps = e.shap_values(X[i1:i2, :])[1]
            shap_list.append(shaps)
        shap_list.append(e.shap_values(X[-1:, :])[1])
        shaps = np.concatenate(shap_list, axis=0)
    
    elif explainer == "GradientSHAP":
        e = shap.GradientExplainer(model, x_background)
        shaps = e.shap_values(x_explanation, nsamples=x_explanation.shape[0])[1]
        
    return shaps
