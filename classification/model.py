import os
import lap
import torch
import pickle
import numpy as np
import math
from torch import nn
from tqdm import tqdm
from prettytable import PrettyTable
import torch.nn.functional as F
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
        
class CNN(nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        # self.filter_sizes1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.filter_sizes1 = list(range(1, 11))
        self.filter_sizes2 = list(range(100, 90, -1))
        self.num_filters = 32                         
        classifier_dropout = 0.1
        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.num_filters, (k, hidden_size)) for k in self.filter_sizes1])
        self.convs2_depthwise = nn.ModuleList([nn.Conv2d(self.num_filters, self.num_filters, (k, 1), groups=self.num_filters) for k in self.filter_sizes2])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes1), num_classes)

    def conv_and_pool(self, x, conv1):
        out = conv1(x)
        out = self.relu(out)
        out = out.squeeze(3)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out
    
    def conv_and_depthwise_conv(self, x, conv1, conv2_depthwise):
        out = conv1(x)
        out = self.relu(out)
        out = conv2_depthwise(out).squeeze(3).squeeze(2)
        return out

    def forward(self, x):
        out = x.float()
        hidden_state = torch.cat([self.conv_and_pool(out, self.convs1[i]) for i in range(len(self.convs1))], 1)
        # hidden_state = torch.cat([self.conv_and_depthwise_conv(out, self.convs1[i], self.convs2_depthwise[i]) for i in range(len(self.convs1))], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out

class CNN_Classifier():
    def __init__(self, max_len=100, n_classes=2, epochs=100, batch_size=32, learning_rate = 0.001, result_save_path = './', hidden_size = 256):
        self.model = CNN(hidden_size)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size = hidden_size
        self.result_save_path = result_save_path + "/" if result_save_path[-1]!="/" else result_save_path
        if not os.path.exists(self.result_save_path): os.makedirs(self.result_save_path)
        # self.best_model_path = os.path.join(result_save_path, "best_model.pt")
        # self.early_stopping = EarlyStopping(patience=5, verbose=True, path=self.best_model_path)

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
        print("val_acc : ",val_acc)
        score_dict = get_score(label, pre)
        print(score_dict)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    def train(self):
        # early_stopping = EarlyStopping(patience=10, verbose=True, path=self.best_model_path)
        train_table = PrettyTable(['typ', 'epo', 'loss', 'TPR', 'TNR', 'Pre', 'Rec', 'F1', 'Acc'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'TPR', 'TNR', 'Pre', 'Rec', 'F1', 'Acc'])
        
        best_metrics = {'F1': 0.0, 'Acc': 0.0, 'Pre': 0.0, 'Rec': 0.0}
        best_model_paths = {'F1': None, 'Acc': None, 'Pre': None, 'Rec': None}
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score])
            print(test_table)
            
            for metric in best_metrics:
                if float(val_score[metric]) > float(best_metrics[metric]):
                    best_metrics[metric] = val_score[metric]
                    model_path = f"{self.result_save_path}/best_model_{metric}.pt"
                    torch.save(self.model.state_dict(), model_path)
                    print(f"Best model saved for {metric}: {best_metrics[metric]}")
            print("")
            
            # early_stopping(val_loss, self.model)
    
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            
    def eval_with_misclassified_indices(self, test_loader):
        print("start evaluating with misclassified indices...")
        self.model.eval()
        losses = []
        predictions = []
        labels = []
        misclassified_indices = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                
                for j, (pred, target) in enumerate(zip(preds, targets)):
                    if pred != target:
                        misclassified_indices.append(i * self.batch_size + j) 

                predictions.extend(preds.cpu().numpy())
                labels.extend(targets.cpu().numpy())
                losses.append(loss.item())

        val_loss = np.mean(losses)
        score_dict = get_score(labels, predictions)
        return val_loss, score_dict, misclassified_indices
    
    def eval_with_classification_details(self, test_loader, test_df):
        self.model.eval()
        losses = []
        predictions = []
        labels = []
        classification_details = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()

                for j, (pred, target) in enumerate(zip(preds, targets)):
                    index = i * self.batch_size + j
                    if pred == target:
                        if pred == 1:
                            classification_details['TP'].append(index)
                        else:
                            classification_details['TN'].append(index)
                    else:
                        if pred == 1:
                            classification_details['FP'].append(index)
                        else:
                            classification_details['FN'].append(index)

                predictions.extend(preds.cpu().numpy())
                labels.extend(targets.cpu().numpy())
                losses.append(loss.item())

        val_loss = np.mean(losses)
        score_dict = get_score(labels, predictions)
        return val_loss, score_dict, classification_details

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
