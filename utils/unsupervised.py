import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def similarity_predict(embeddings, topk=1, distance='cosine_distance'):
    """
    Return true labels and predicted labels. If topk > 1 then the function
    will return topk number of closest answers to question
    
    embeddings:
    type: a list of dictionaries, where each dictionary contains 
          correct answer and embeddings for question and answers
    ------------------------------------------------------------
    
    topk: number of closest answers to return
    ------------------------------------------------------------
    
    distance: type of distance
    avaliable values: 'cosine_distance', 'cosine_similarity'
    """
    y_pred, y = [], []
    num = re.compile('[0-9]+')
    for i in tqdm(range(len(embeddings))):
        # prediction
        distances = np.zeros(4)
        for j in range(1, 5):
            if distance == 'cosine_similarity':    
                distances[j - 1] = cosine_similarity(embeddings[i]['q'].reshape(1, -1), 
                                                 embeddings[i][f'a{j}'].reshape(1, -1))[0][0]
            elif distance == 'cosine_distance':
                distances[j - 1] = cosine_distances(embeddings[i]['q'].reshape(1, -1), 
                                                 embeddings[i][f'a{j}'].reshape(1, -1))[0][0]
        if distance == 'cosine_similarity':
            if topk == 1:
                y_pred.append(np.argmax(distances))
            else:
                y_pred.append(np.argsort(distances)[::-1][:topk])
        elif distance == 'cosine_distance':
            if topk == 1:
                y_pred.append(np.argmin(distances))
            else:
                y_pred.append(np.argsort(distances)[:topk])
        else:
            raise ValueError(f'{distance} is not supported')
        
        
        # true labels
        if isinstance(embeddings[i]['ca'], int):
            y.append(int(embeddings[i]['ca']) - 1)
        else:
            y.append(int(num.findall(embeddings[i]['ca'])[0]) - 1)
    
    
    return y, y_pred



def topk_accuracy(y, y_pred, k=2):
    """
    Returns ration of cases in which the correct answer 
    is in k closest
    """
    pred = []
    for i in range(len(y)):
        if y[i] in y_pred[i][:k]:
            pred.append(1)
        else:
            pred.append(0)
    return sum(pred) / len(pred)