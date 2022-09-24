#!/usr/bin/env python
# coding: utf-8

# # Initialisation

# In[1]:


import torch
import torch.nn as nn
import torchvision
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" #Use CPU


# # Data

# In[31]:


Seed = 321
batch_size = 1000
C1, H1, W1, K = 3, 32, 32, 10

import ssl #Fixes: CERTIFICATE_VERIFY_FAILED
ssl._create_default_https_context = ssl._create_unverified_context
train_ds = torchvision.datasets.CIFAR10(root = 'Data', train = True, download = True, transform = torchvision.transforms.ToTensor())
test_ds = torchvision.datasets.CIFAR10(root = 'Data', train = False, download = True, transform = torchvision.transforms.ToTensor())
classes = train_ds.classes
train = torch.utils.data.DataLoader(dataset = train_ds, batch_size = batch_size, shuffle = True)
test = torch.utils.data.DataLoader(dataset = test_ds, batch_size = batch_size, shuffle = True)


# # Model

# ## Architecture

# In[32]:


class NeuralNet(nn.Module):
    def __init__(self, relu_param, elu_param, H1, W1, K, kernel_size, stride, padding, dilation, pool, pool_dilation, pool_stride, nonlinearities, widths, has_bias, dropout):
        super(NeuralNet, self).__init__()
        self.relu_param = relu_param
        self.elu_param = elu_param
        self.H1 = H1
        self.W1 = W1
        self.K = K
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.pool_dilation = pool_dilation
        self.pool_stride = pool_stride
        self.nonlinearities = nonlinearities
        self.widths = widths
        self.has_bias = has_bias
        self.dropout = dropout
        
        self.CNN = nn.ModuleList([
            nn.Conv2d(in_channels = self.widths[0], out_channels = self.widths[1], kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, dilation = self.dilation, bias = self.has_bias[0]),
            self.apply_nonlinearity(0),
            nn.Dropout(p = self.dropout[0], inplace = False),
            nn.MaxPool2d(kernel_size = self.pool, dilation = self.pool_dilation, stride = self.pool_stride),
            
            nn.Conv2d(in_channels = self.widths[1], out_channels = self.widths[1], kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, dilation = self.dilation, bias = self.has_bias[0]),
            self.apply_nonlinearity(0),
            nn.Dropout(p = self.dropout[0], inplace = False),
            nn.MaxPool2d(kernel_size = self.pool, dilation = self.pool_dilation, stride = self.pool_stride)
        ])
        
        self.FFNN = nn.ModuleList([
            nn.Linear(in_features = int(((self.H1/self.pool)/2) * ((self.W1/self.pool)/2) * widths[1]), out_features = self.widths[2], bias = self.has_bias[1]),
            self.apply_nonlinearity(1),
            nn.Dropout(p = self.dropout[1], inplace = False),
            
            nn.Linear(in_features = self.widths[2], out_features = self.widths[3], bias = self.has_bias[2]),
            self.apply_nonlinearity(2),
            nn.Dropout(p = self.dropout[2], inplace = False),
            
            nn.Linear(in_features = self.widths[3], out_features = self.widths[4], bias = self.has_bias[3]),
            self.apply_nonlinearity(3),
            nn.Dropout(p = self.dropout[3], inplace = False),
        ])
        
        self.output = nn.Linear(in_features = self.widths[4], out_features = self.K, bias = self.has_bias[4])
        
    def apply_nonlinearity(self, layer):
        res = None
        if (self.nonlinearities[layer] == "relu"):
            res = nn.ReLU() if (self.relu_param == 0) else self.layers.append(nn.LeakyReLU(negative_slope = self.relu_param))
        elif (self.nonlinearities[layer] == "tanh"):
            res = nn.Tanh()
        elif (self.nonlinearities[layer] == "relu6"):
            res = nn.ReLU6()
        elif (self.nonlinearities[layer] == "sigmoid"):
            res = nn.Sigmoid()
        elif (self.nonlinearities[layer] == "tanh"):
            res = nn.Tanh()
        elif (self.nonlinearities[layer] == "elu"):
            res = nn.ELU(alpha = self.elu_param)
        return res
        
    def forward(self, x):
        for i in range(8):
            x = self.CNN[i](x)
        x = x.view(x.shape[0], -1)
        for i in range(9):
            x = self.FFNN[i](x)
        x = self.output(x)
        
        return x


# In[47]:


def gradient_descent(model, criterion, optimiser, train, test, iterations):
    model = model.to(device)
    train_losses, test_losses = np.zeros(iterations), np.zeros(iterations)
    
    for iteration in range(iterations):
        train_loss = []
        
        for inputs, targets in train:
            inputs = inputs.to(device)
            targets = targets.to(device).long()            
            model.train()
            optimiser.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
            
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        
        test_loss, test_accuracy, test_auc, test_f1 = [], [], [], []
        for inputs, targets in test:
            inputs = inputs.to(device)
            targets = targets.to(device).long()            
            
            model.eval()
            with torch.no_grad():
                y_probs = model(inputs)

                cur_test_loss = criterion(y_probs, targets).item()
                y_probs = nn.Softmax(dim = 1)(y_probs)
                _, y_hat = torch.max(y_probs, 1)
                y_probs = y_probs.cpu().numpy()
                y_hat = y_hat.cpu().numpy()
                targets = targets.cpu().numpy()

                cur_test_accuracy = np.mean(y_hat == targets)
                cur_test_auc = roc_auc_score(targets, y_probs, multi_class = "ovr", average = 'weighted')
                cur_test_f1 = f1_score(targets, y_hat, average = 'weighted')
            
            test_loss.append(cur_test_loss)
            test_accuracy.append(cur_test_accuracy)
            test_auc.append(cur_test_auc)
            test_f1.append(cur_test_f1)
        test_loss =  np.mean(test_loss)
        test_accuracy = np.mean(test_accuracy)
        test_auc = np.mean(test_auc)
        test_f1 = np.mean(test_f1)

        train_losses[iteration] = train_loss
        test_losses[iteration] = test_loss
        
        if (iteration + 1) % 1 == 0 or iteration == 0 or iteration == iterations - 1:
            print(f'Iteration {iteration + 1}/{iterations}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f} | Acc: {test_accuracy:.3f}, AUC: {test_auc:.3f}, F1: {test_f1:.3f}')
    
    return train_losses, test_losses


# ## Hyper Parameters

# In[60]:


kernel_size = 3
stride = 1
dilation = 1
padding = 1
pool = 2
pool_dilation = 1
pool_stride = pool
relu_param = 0
elu_param = 0.8
widths = [C1, 25, 64, 32, 16]
nonlinearities = ["relu"] * 4
dropout = [0.0] * 4
has_bias = [True] * 5
l2_lamda = 0.0
mu = 0.9


# In[61]:


train_losses = np.array([])
test_losses = np.array([])
criterion = nn.CrossEntropyLoss()
model = NeuralNet(relu_param, elu_param, H1, W1, K, kernel_size, stride, padding, dilation, pool, pool_dilation, pool_stride, nonlinearities, widths, has_bias, dropout).to(device)
model


# In[62]:


learning_rate = 1e-2
optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (mu, 0.999), weight_decay = l2_lamda, amsgrad = False)


# In[64]:


iterations = 20
new_train_losses, new_test_losses = gradient_descent(model, criterion, optimiser, train, test, iterations)
train_losses = np.append(train_losses, new_train_losses)
test_losses = np.append(test_losses, new_test_losses)


# In[65]:


plt.plot(train_losses, label = f"Train")
plt.plot(test_losses, label = f"Test")
plt.legend()
plt.show()


# In[66]:


model.eval()
with torch.no_grad():
    model = model.to(device)
    
    Preds_prob = []
    Preds = []
    Targets = []
    
    for inputs, targets in test:
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        
        model.eval()
        with torch.no_grad():
            outputs_prob = model(inputs)
            outputs_prob = nn.Softmax(dim = 1)(outputs_prob)
            _, outputs = torch.max(outputs_prob, 1)
        
        Preds_prob.append(outputs_prob)
        Preds.append(outputs)
        Targets.append(targets)
        
    Preds_prob = torch.cat(Preds_prob).cpu().numpy()
    Preds = torch.cat(Preds).cpu().numpy()
    Targets = torch.cat(Targets).cpu().numpy()
    
    test_Acc = np.mean(Preds == Targets)
    test_AUC = roc_auc_score(Targets, Preds_prob, multi_class = "ovr", average = 'weighted')
    test_F1 = f1_score(Targets, Preds, average = 'weighted')
    print(f'Accuracy: {test_Acc * 100:.3f}%. AUC: {test_AUC:.3f}, F1: {test_F1:.3f}')
    
    ConfusionMatrixDisplay(confusion_matrix(Targets, Preds), display_labels = classes).plot()
    plt.show()


# # Hyperparameters 2

# In[67]:


kernel_size = 3
stride = 1
dilation = 1
padding = 1
pool = 2
pool_dilation = 1
pool_stride = pool
relu_param = 0
elu_param = 0.8
widths = [C1, 25, 64, 32, 16]
nonlinearities = ["relu"] * 4
dropout = [0.3] * 4
has_bias = [True] * 5
l2_lamda = 0.05
mu = 0.9


# In[68]:


train_losses = np.array([])
test_losses = np.array([])
criterion = nn.CrossEntropyLoss()
model = NeuralNet(relu_param, elu_param, H1, W1, K, kernel_size, stride, padding, dilation, pool, pool_dilation, pool_stride, nonlinearities, widths, has_bias, dropout).to(device)
model


# In[69]:


learning_rate = 1e-2
optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (mu, 0.999), weight_decay = l2_lamda, amsgrad = False)


# In[82]:


iterations = 100
new_train_losses, new_test_losses = gradient_descent(model, criterion, optimiser, train, test, iterations)
train_losses = np.append(train_losses, new_train_losses)
test_losses = np.append(test_losses, new_test_losses)


# In[83]:


plt.plot(train_losses, label = f"Train")
plt.plot(test_losses, label = f"Validation")
plt.legend()
plt.show()


# In[84]:


model.eval()
with torch.no_grad():
    model = model.to(device)
    
    Preds_prob = []
    Preds = []
    Targets = []
    
    for inputs, targets in test:
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        
        model.eval()
        with torch.no_grad():
            outputs_prob = model(inputs)
            outputs_prob = nn.Softmax(dim = 1)(outputs_prob)
            _, outputs = torch.max(outputs_prob, 1)
        
        Preds_prob.append(outputs_prob)
        Preds.append(outputs)
        Targets.append(targets)
    
    Preds_prob = torch.cat(Preds_prob).cpu().numpy()
    Preds = torch.cat(Preds).cpu().numpy()
    Targets = torch.cat(Targets).cpu().numpy()
    
    test_Acc = np.mean(Preds == Targets)
    test_AUC = roc_auc_score(Targets, Preds_prob, multi_class = "ovr", average = 'weighted')
    test_F1 = f1_score(Targets, Preds, average = 'weighted')
    print(f'Accuracy: {test_Acc * 100:.3f}%. AUC: {test_AUC:.3f}, F1: {test_F1:.3f}')
    
    ConfusionMatrixDisplay(confusion_matrix(Targets, Preds), display_labels = classes).plot()
    plt.show()


# # Hyperparameters 3

# In[86]:


kernel_size = 3
stride = 1
dilation = 1
padding = 1
pool = 2
pool_dilation = 1
pool_stride = pool
relu_param = 0
elu_param = 0.8
widths = [C1, 32, 32, 16, 16]
nonlinearities = ["relu", "tanh", "tanh", "tanh"]
dropout = [0.1] * 4
has_bias = [True] * 5
l2_lamda = 0.005
mu = 0.9


# In[87]:


train_losses = np.array([])
test_losses = np.array([])
criterion = nn.CrossEntropyLoss()
model = NeuralNet(relu_param, elu_param, H1, W1, K, kernel_size, stride, padding, dilation, pool, pool_dilation, pool_stride, nonlinearities, widths, has_bias, dropout).to(device)
model


# In[88]:


learning_rate = 1e-2
optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (mu, 0.999), weight_decay = l2_lamda, amsgrad = False)


# In[92]:


iterations = 20
new_train_losses, new_test_losses = gradient_descent(model, criterion, optimiser, train, test, iterations)
train_losses = np.append(train_losses, new_train_losses)
test_losses = np.append(test_losses, new_test_losses)


# In[95]:


plt.plot(train_losses, label = f"Train")
plt.plot(test_losses, label = f"Validation")
plt.legend()
plt.show()


# In[96]:


model.eval()
with torch.no_grad():
    model = model.to(device)
    
    Preds_prob = []
    Preds = []
    Targets = []
    
    for inputs, targets in test:
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        
        model.eval()
        with torch.no_grad():
            outputs_prob = model(inputs)
            outputs_prob = nn.Softmax(dim = 1)(outputs_prob)
            _, outputs = torch.max(outputs_prob, 1)
        
        Preds_prob.append(outputs_prob)
        Preds.append(outputs)
        Targets.append(targets)
    
    Preds_prob = torch.cat(Preds_prob).cpu().numpy()
    Preds = torch.cat(Preds).cpu().numpy()
    Targets = torch.cat(Targets).cpu().numpy()
    
    test_Acc = np.mean(Preds == Targets)
    test_AUC = roc_auc_score(Targets, Preds_prob, multi_class = "ovr", average = 'weighted')
    test_F1 = f1_score(Targets, Preds, average = 'weighted')
    print(f'Accuracy: {test_Acc * 100:.3f}%. AUC: {test_AUC:.3f}, F1: {test_F1:.3f}')
    
    ConfusionMatrixDisplay(confusion_matrix(Targets, Preds), display_labels = classes).plot()
    plt.show()

