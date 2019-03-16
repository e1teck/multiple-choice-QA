import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def prepare_for_supervised(embeddings):
    """
    Returns the list of the concatenated pairs of question
    and answers and the list of target labels
    
    embeddigns: 
    type: a list of dictionaries, where each dictionary contains 
          correct answer and embeddings for question and answers
    """
    
    X, y = [], []
    num = re.compile('[0-4]+')
    for problem in tqdm(embeddings):
        pairs = [np.concatenate((problem['q'], problem[i])) for i in ('a1', 'a2', 'a3', 'a4')]
        if isinstance(problem['ca'], int):
            correct_answer = problem['ca'] - 1
        else:   
            correct_answer = int(num.findall(problem['ca'])[0])
            if correct_answer != 0:
                correct_answer -= 1
        
        labels = [0, 0, 0, 0]
        labels[correct_answer] = 1
        X.extend(pairs)
        y.extend(labels)
    return X, y



def make_pairs(dct):
    pairs = []
    for i in range(1, 5):
        pairs.append(np.concatenate((dct['q'], dct[f'a{i}'])))
    return pairs

# prediction of the answer to the given question
def predict_probs(elmo_embs, model, topk=1, predict_proba=True):
    pred = []
    for problem in tqdm(elmo_embs):
        pairs = make_pairs(problem)
        if predict_proba:
            probs = model.predict_proba(pairs)[:, 1]
        else:
            probs = model.decision_function(pairs)
        if topk == 1:
            pred.append(np.argmax(probs))
        else:
            pred.append(np.argsort(probs)[::-1][:topk])
    return pred

