import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys

seed = 1234
np.random.seed(seed)

def stratify_split(data, split_ratio: list):
    '''
    train/val/test dataset generation, stratified split by outcome
    
    ---
    Params:
    data: df, data with (binary) outcome named by "label"
    split_ratio: list, indicating train: validation: test
    
    ---
    Return:
    train_data, val_data, test_data
    '''
    seed = 1234
    np.random.seed(seed)
    
    # data = df.reindex(np.random.permutation(df.index))
    split_ratio = split_ratio / np.sum(split_ratio)
    cum_split_ratio = np.cumsum(split_ratio)
    index_0, index_1 = data.index[data["label"] == 0], data.index[data["label"] == 1]
    num_0, num_1 = len(index_0), len(index_1)
    
    event_rate = round(np.mean(data["label"] == 1), 3)  
    num_1 = int(num_0 * (event_rate/ (1 - event_rate)))
    index_1 = np.random.choice(index_1, num_1, replace=False)

    train_data_0, val_data_0, test_data_0 = np.split(
        data.loc[index_0], [int(cum_split_ratio[0]*num_0), int(cum_split_ratio[1]*num_0)])
    train_data_1, val_data_1, test_data_1 = np.split(
        data.loc[index_1], [int(cum_split_ratio[0]*num_1), int(cum_split_ratio[1]*num_1)])

    train_data = pd.concat([train_data_0, train_data_1])
    val_data = pd.concat([val_data_0, val_data_1])
    test_data = pd.concat([test_data_0, test_data_1])

    return (train_data, val_data, test_data)
    
def generate_background(data, bg_size=1000, bg_mor=0.5, re=10):
    '''
    Generating background data with specific minority-majority ratio
    ---
    Params:
    data: data frame, data for generating backgound data
    bg_size: the size of background data
    bg_mor: float, background minority-overall rate
    re: number of repetition
    '''
    # seed = 1234
    # np.random.seed(seed)

    event_rate = round(np.mean(data["label"] == 1), 3)
    if event_rate < 0.5:
        minority_rate = event_rate 
        mn_idx = data.index[data["label"] == 1]
        mj_idx = data.index[data["label"] == 0]
    elif event_rate == 0.5:
        return(np.random.choice(data, bg_size, replace=False))
    else: 
        minority_rate = (1 - event_rate)
        mn_idx = data.index[data["label"] == 0]
        mj_idx = data.index[data["label"] == 1]
         
    bg_num_0 = int(bg_size * (1 - bg_mor))
    bg_num_1 = int(bg_size * bg_mor)
    if bg_num_1 > len(mj_idx):
        sys.exit("Please use smaller size of background OR smaller background minority-majority ratio")# 0 for majority
    
    for i in range(re):
        bg_data_0 = data.loc[np.random.choice(mj_idx, bg_num_0, replace=False)]
        bg_data_1 = data.loc[np.random.choice(mn_idx, bg_num_1, replace=False)]
        background_data = pd.concat([bg_data_0, bg_data_1])
            
    return background_data

def calculate_WSS(data, kmax, show=True):
    '''
    calculate Within-Cluster-Sum of Squared Errors (WSS)
    
    ---
    Params:
    data: data for clustering
    kmax: maxmium of clusters  
    show: bool, whether to show the plot
    '''
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(data)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(data.shape[0]):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.sum((data.iloc[i, ] - curr_center) ** 2)

        sse.append(curr_sse)
    
    # elbow plot
    plt.plot(np.arange(kmax)+1, sse)
    plt.plot(np.arange(kmax)+1, sse, "bo-")  
    if show:
        plt.show()   
        
    return sse
  
    
def undersample_kmeans(data, mor=0.5, n_clusters=3):
    '''
    Generate K-Means-based balanced data (binary outcome) 
    ----
    Params:
    data: data frame with (binary) label named as "label"
    mor: float, targeted minority-overall rate, default - 0.5
    n_clusters: number of clusters for KMeans, default - 3
    '''
    seed = 1234
    np.random.seed(seed)
    
    event_rate = round(np.mean(data["label"] == 1), 3)
    minority_label = 1 if event_rate < 0.5 else 0
    majority_label = 1 - minority_label
    
    # print(majority_label)
    majority = data.loc[data["label"] == majority_label, :]
    minority = data.loc[data["label"] == minority_label, :]
    num_minority = minority.shape[0]
    majority_size = num_minority / mor * (1 - mor) 

    estimator = KMeans(n_clusters=n_clusters, random_state=seed)
    estimator.fit(data.loc[data["label"] == majority_label, :].drop(columns="label"))
    labels_pred = estimator.labels_
    majority.loc[:, "label_pred"] = labels_pred
    print(np.unique(labels_pred, return_counts=True))

    def select(label=0):
        return np.random.choice(majority.index[majority["label_pred"] == label], int(np.min([majority_size // n_clusters,  len(majority.index[majority["label_pred"] == label])])), replace=False)

    us_data = pd.concat(
        [data.loc[np.concatenate([select(k) for k in range(n_clusters)]), :], minority])
    
    return us_data

if __name__ == "__main__":
    data = pd.read_csv("../data/data.csv")
    event_rate = sum(data["label"] == 1) / data.shape[0]
    mor = event_rate
    
    train_data, val_data, test_data = stratify_split(data, [6, 2, 2])
    
    bg_size = 1000
    bg_mor = 0.5 # [mor, 0.2, 0.3, 0.4, 0.5]
    background_data = generate_background(data, bg_size, bg_mor)
    print(background_data.columns)
    
    n_vars = val_data.shape[1] - 1
    kmax = 7
    sse = calculate_WSS(val_data.loc[val_data["label"] == 0, ].drop(columns="label"), kmax)
    
    explanation_data = undersample_kmeans(val_data, 0.5, n_clusters = 3)
