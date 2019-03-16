import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import re

def make_siamese_pairs(dct):
    pairs = []
    num = re.compile('[0-4]+')
    for i in range(1, 5):
        pairs.append((dct['q'], dct[f'a{i}']))
        if isinstance(dct['ca'], int):
            correct_answer = dct['ca'] - 1
        else:   
            correct_answer = int(num.findall(dct['ca'])[0])
        if correct_answer != 0:
            correct_answer -= 1
    target = [0, 0, 0, 0]
    target[correct_answer] = 1
    return pairs, target


def prepare_for_siamese(embeds):
    X, y = [], []
    for problem in embeds:
        temp = make_siamese_pairs(problem)
        X.extend(temp[0])
        y.extend(temp[1])
    return X, y


def get_batches(train_x, train_y, batch_size=64):
    for i in range(0, len(train_x) - batch_size, batch_size):
        q = tuple([i[0].view(1, -1) for i in train_x[i:i+batch_size]])
        a = tuple([i[1].view(1, -1) for i in train_x[i:i+batch_size]])
        question = torch.cat(q, 0)
        answer = torch.cat(a, 0)
        yield (question, answer, torch.from_numpy(train_y[i:i+batch_size]))
        
        

        
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.question = nn.Sequential(nn.Linear(1024, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU())
        self.answer = nn.Sequential(nn.Linear(1024, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 1))
        
    def forward(self, q, a):
        question = self.question(q)
        answer = self.answer(a)
        x = torch.cat((question, answer), 1)
        out = self.classifier(x)
        
        return out

    
    
def siamese_pred(test, model):
    prediction = []
    for problem in test:
        pair, _ = make_siamese_pairs(problem)
        pair = [(torch.from_numpy(i[0]).cuda().view(1, -1), 
                 torch.from_numpy(i[1]).cuda().view(1, -1)) for i in pair]
        question = torch.cat(tuple([i[0] for i in pair]), 0)
        answers = torch.cat(tuple([i[1] for i in pair]), 0)
    
        preds = model(question, answers).squeeze().cpu().detach().numpy()
        prediction.append(np.argmax(preds))
    return prediction