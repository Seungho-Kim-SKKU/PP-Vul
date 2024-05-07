import os
import lap
import torch
import numpy
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import sys

def get_accuracy(labels, prediction):    
    cm = confusion_matrix(labels, prediction)
    def linear_assignment(cost_matrix):    
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]    
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy 

def get_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, predictions, average='macro')
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "TPR": format(tpr * 100, '.3f'),
        "TNR": format(tnr * 100, '.3f'),
        "Pre": format(precision * 100, '.3f'),
        "Rec": format(recall * 100, '.3f'),        
        "F1" : format(f1 * 100, '.3f'),
        "Acc"  : format(accuracy * 100, '.3f')
    }

class Dataset(Dataset):
    def __init__(self, texts, targets, max_len, hidden_size):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.hidden_size = hidden_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = self.targets[idx]
        vectors = np.zeros(shape=(1,self.max_len,self.hidden_size))        
        for i in range(min(len(feature) - 1, self.max_len - 1)):
            vectors[0][i + 1] = feature[i + 1]
        return {
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long)
        }
    
class Flatten(torch.nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)
    
class Cube(torch.nn.Module):
    def forward(self, x):
        return x**3

class CNN(nn.Module):
    def __init__(self, hidden_size = 256, num_classes = 2):
        super(CNN, self).__init__()
        self.filter_sizes1 = 4
        self.filter_sizes2 = 65 - self.filter_sizes1
        self.num_filters = 16                  
        self.classifier_dropout = 0.1
        self.Conv1 = nn.Conv2d(1, self.num_filters, (self.filter_sizes1, hidden_size))
        self.Cube = Cube()
        self.Conv2_depthwise = nn.Conv2d(self.num_filters, self.num_filters, (self.filter_sizes2, 1), groups=self.num_filters)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.Flatten = Flatten()
        self.FC = nn.Linear(self.num_filters, num_classes)        

    def forward(self, x):
        out = x.float()
        out = self.Conv1(out)
        out = self.Cube(out)
        out = self.Conv2_depthwise(out)
        out = self.dropout(out)
        out = self.Flatten(out).squeeze(1)
        out = self.FC(out)
        return out

class CNN_Classifier():
    def __init__(self, max_len=64, n_classes=2, epochs=100, batch_size=32, learning_rate = 0.001, hidden_size = 256):
        self.model = CNN(hidden_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size = hidden_size

    def preparation(self, X_train, y_train, X_valid, y_valid):
        self.train_set = Dataset(X_train, y_train, self.max_len, self.hidden_size)
        self.valid_set = Dataset(X_valid, y_valid, self.max_len, self.hidden_size)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs  = self.model( vectors )
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()           
            
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))   
            labels += list(np.array(targets.cpu()))      

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_score(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for _, data in progress_bar:
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print(f"val_acc :", 100 * val_acc.item())
        score_dict = get_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    
    def train(self):
        train_table = PrettyTable(['typ', 'epo', 'loss', 'TPR', 'TNR', 'Pre', 'Rec', 'F1', 'Acc'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'TPR', 'TNR', 'Pre', 'Rec', 'F1', 'Acc'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score])
            print(test_table)
            print("\n")
