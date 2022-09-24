#!/usr/bin/env python
# coding: utf-8

# # Initialisation

# In[2]:


import torchvision

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

def y_probs(IndexVector, nCols):
    Result = np.zeros((len(IndexVector), nCols))
    for i in range(len(IndexVector)):
        Result[i, int(IndexVector[i])] = 1
    return Result


# # Data

# In[17]:


Seed = 111 #For reproducibility

train_dataset = torchvision.datasets.FashionMNIST(root = 'Data', train = True, download = True)

test_dataset = torchvision.datasets.FashionMNIST(root = 'Data', train = False, download = True)

ClassNames = train_dataset.classes           
x_train = train_dataset.data.cpu().numpy()   
y_train = train_dataset.targets.cpu().numpy()
x_test = test_dataset.data.cpu().numpy()     
y_test = test_dataset.targets.cpu().numpy()  

N, H1, W1 = x_train.shape
C1 = 1 #One colour channel
K = 10
D = H1 * W1 #The dimensionality of the dataset
print(f"N: {N}, C1: {C1}, H1: {H1}, W1: {W1}, K: {K}")
print("Class names:", ClassNames)
print(f"\nBefore scaling: Min ({x_train.min()}),   Max ({x_train.max()}), Data type ({x_train.dtype})")

#Original data goes from 0 to 255 but we're scaling it
x_train = (x_train / 255).astype(np.float32)
x_test = (x_test / 255).astype(np.float32)
print(f" After scaling: Min ({x_train.min()}), Max ({x_train.max()}), Data type ({x_train.dtype})")

print(f"\nX_Train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape : {x_test.shape}, y_test shape : {y_test.shape}")

sns.countplot(y_train)
plt.show()


# In[4]:


orig = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [8.5, 5.5]
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5)
ax1.imshow(x_train[np.argmax(y_train == 0)], cmap = 'gray')
ax1.set_title(ClassNames[0])
ax1.axis('off')
ax2.imshow(x_train[np.argmax(y_train == 1)], cmap = 'gray')
ax2.set_title(ClassNames[1])
ax2.axis('off')
ax3.imshow(x_train[np.argmax(y_train == 2)], cmap = 'gray')
ax3.set_title(ClassNames[2])
ax3.axis('off')
ax4.imshow(x_train[np.argmax(y_train == 3)], cmap = 'gray')
ax4.set_title(ClassNames[3])
ax4.axis('off')
ax5.imshow(x_train[np.argmax(y_train == 4)], cmap = 'gray')
ax5.set_title(ClassNames[4])
ax5.axis('off')
ax6.imshow(x_train[np.argmax(y_train == 5)], cmap = 'gray')
ax6.set_title(ClassNames[5])
ax6.axis('off')
ax7.imshow(x_train[np.argmax(y_train == 6)], cmap = 'gray')
ax7.set_title(ClassNames[6])
ax7.axis('off')
ax8.imshow(x_train[np.argmax(y_train == 7)], cmap = 'gray')
ax8.set_title(ClassNames[7])
ax8.axis('off')
ax9.imshow(x_train[np.argmax(y_train == 8)], cmap = 'gray')
ax9.set_title(ClassNames[8])
ax9.axis('off')
ax10.imshow(x_train[np.argmax(y_train == 9)], cmap = 'gray')
ax10.set_title(ClassNames[9])
ax10.axis('off')
plt.show()
plt.rcParams['figure.figsize'] = orig


# # Functions

# # Neural Network

# ## Architecture

# In[5]:


class NeuralNet():    
    def __init__(self, nonlinearitys, size, widths):
        self.nonlinearitys = nonlinearitys
        self.size = size
        self.widths = widths
        self.layers = []
        
        old_size = self.size
        for layer in range(len(self.widths)):
            size = self.widths[layer]
            self.layers.append(linear(nonlinearity = self.nonlinearitys[layer], dkernel = None, dbias = None, gradkernel = None, gradbias = None, kernel = np.random.normal(size = (old_size, size)) / np.sqrt(old_size), bias = np.random.normal(size = size)))
            old_size = size
    
    def predict(self, x):
        res = x.reshape(len(x), -1)
        for i in range(len(self.widths)):
            res = self.layers[i].predict(res)
        return res
    
    def backprop(self, y, y_hat, lr):
        self.layers[-1].dkernel = self.layers[-1].d_out_kernel(y, y_hat, self.widths[-1])
        self.layers[-1].dbias = self.layers[-1].d_out_bias()
        self.layers[-1].gradkernel = self.layers[-1].grad_kernel()
        self.layers[-1].gradbias = self.layers[-1].grad_bias()
        self.layers[-1].take_step(lr)
        
        if len(self.widths) > 1:
            for layer in range(len(self.widths) - 2, -1, -1):
                self.layers[layer].dkernel = self.layers[layer].d_kernel(self.layers[layer+1].dkernel, self.layers[layer+1].kernel, self.layers[layer+1].x, self.layers[layer].nonlinearity)
                self.layers[layer].dbias = self.layers[layer].d_bias()
                self.layers[layer].gradkernel = self.layers[layer].grad_kernel()
                self.layers[layer].gradbias = self.layers[layer].grad_bias()
                self.layers[layer].take_step(lr)
    
    def train(self):
        for layer in range(len(self.widths)):
            self.layers[layer].train()
    
    def evaluate(self):
        for layer in range(len(self.widths)):
            self.layers[layer].evaluate()
    
    def __len__(self):
        return len(self.widths)


# In[6]:


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a), axis = 1, keepdims = True)

def relu(a):
    return a * (a > 0)

def sigmoid(a):
    if np.all(a >= 0):
        return 1 / (1 + np.exp(-a))
    else:
        return np.exp(a) / (1 + np.exp(a))

def loss(y, y_hat):
    return -np.mean(y * np.log(y_hat))

class linear():    
    def __init__(self, nonlinearity, dkernel, dbias, gradkernel, gradbias, kernel, bias):
        self.nonlinearity = nonlinearity
        self.dkernel = dkernel
        self.dbias = dbias
        self.gradkernel = gradkernel
        self.gradbias = gradbias
        self.kernel = kernel
        self.bias = bias
        self.x = None
        self.is_training = False
        
    def evaluate(self):
        self.is_training = False
    
    def train(self):
        self.is_training = True
        
    def take_step(self, lr):
        self.kernel = self.kernel - (lr * self.gradkernel)
        self.bias = self.bias - (lr * self.gradbias)
        
    def predict(self, X):
        if self.is_training:
            self.x = X.copy()
        Res = X.dot(self.kernel) + self.bias
        Res = sigmoid(Res) if self.nonlinearity.lower() == "sigmoid" else relu(Res) if self.nonlinearity.lower() == "relu" else softmax(Res)
        return Res
    
    def grad_relu(self, x):
        return np.sign(x)
        
    def grad_sigmoid(self, x):
        return x * (1 - x)
        
    def grad_kernel(self):
        return self.x.T.dot(self.dkernel)
        
    def grad_bias(self):
        return self.dbias.copy()
    
    def d_kernel(self, Delta, kernel, x, nonlinearity):
        return Delta.dot(kernel.T) * (self.grad_sigmoid(x) if nonlinearity.lower() == "sigmoid" else self.grad_relu(x))
    
    def d_bias(self):
        return np.sum(self.dkernel.copy(), axis = 0)
    
    def d_out_kernel(self, y, y_hat, K):
        return y_hat - y_probs(y, K)
        
    def d_out_bias(self):
        return np.sum(self.dkernel.copy(), axis = 0)


# ## Hyperparameters 1

# ### Neural Network Model

# In[17]:


nonlinearities = ["sigmoid", "sigmoid"] + ["softmax"] #Last is softmax for multiclass classification
widths = [140, 80, K]
model = NeuralNet(nonlinearities, D, widths)


# ### Optimisation

# In[18]:


iterations = 300
lr = 1e-5
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []


# In[19]:


for iteration in range(iterations):
    model.train()
    
    y_prob = model.predict(x_train)
    y_hat = np.argmax(y_prob, axis = 1)
    actual_prob = y_probs(y_train, K)
    
    loss_train = loss(actual_prob, y_prob)
    accuracy_train = np.mean(y_hat == y_train)
    
    model.backprop(y_train, y_prob, lr)
    model.evaluate()
    
    y_prob_test = model.predict(x_test)
    y_hat_test = np.argmax(y_prob_test, axis = 1)
    actual_test_prob = y_probs(y_test, K)
    
    loss_test = loss(actual_test_prob, y_prob_test)
    accuracy_test = np.mean(y_hat_test == y_test)
    
    train_losses.append(loss_train)
    train_accuracies.append(accuracy_train)
    
    test_losses.append(loss_test)
    test_accuracies.append(accuracy_test)
    
    if (iteration + 1) % 1 == 0 or iteration == iterations - 1 or iteration == 0:
        print(f'Iteration {iteration + 1}/{iterations}. Train loss: {loss_train:.4f}, Train acc: {accuracy_train:.2f} - Test loss: {loss_test:.4f}, Test acc: {accuracy_test:.2f}')


# ### Evaluation

# In[20]:


#Plotting the metrics for Train and Test sets
orig = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [15, 5]
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Loss")
ax1.plot(train_losses, label = f"Train Loss")
ax1.plot(test_losses, label = f"Validation Loss")
ax1.legend()
ax2.set_title("Accuracy")
ax2.plot(train_accuracies, label = f"Train Acc")
ax2.plot(test_accuracies, label = f"Validation Acc")
ax2.legend()
plt.show()
plt.rcParams['figure.figsize'] = orig

#Plotting the confusion matrix
orig = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [10, 5]
ConfusionMatrixDisplay(confusion_matrix(y_test, y_hat_test), display_labels = ClassNames).plot()
plt.show()
plt.rcParams['figure.figsize'] = orig


# ## Hyperparameters 2

# ### Neural Network Model

# In[7]:


nonlinearities = ["relu", "relu"] + ["softmax"] #Last is softmax for multiclass classification
widths = [140, 80, K]
model = NeuralNet(nonlinearities, D, widths)


# ### Optimisation

# In[8]:


iterations = 300
lr = 6e-7
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []


# In[9]:


for iteration in range(iterations):
    model.train()
    
    y_prob = model.predict(x_train)
    y_hat = np.argmax(y_prob, axis = 1)
    actual_prob = y_probs(y_train, K)
    
    loss_train = loss(actual_prob, y_prob)
    accuracy_train = np.mean(y_hat == y_train)
    
    model.backprop(y_train, y_prob, lr)
    model.evaluate()
    
    y_prob_test = model.predict(x_test)
    y_hat_test = np.argmax(y_prob_test, axis = 1)
    actual_test_prob = y_probs(y_test, K)
    
    loss_test = loss(actual_test_prob, y_prob_test)
    accuracy_test = np.mean(y_hat_test == y_test)
    
    train_losses.append(loss_train)
    train_accuracies.append(accuracy_train)
    
    test_losses.append(loss_test)
    test_accuracies.append(accuracy_test)
    
    if (iteration + 1) % 1 == 0 or iteration == iterations - 1 or iteration == 0:
        print(f'Iteration {iteration + 1}/{iterations}. Train loss: {loss_train:.4f}, Train acc: {accuracy_train:.2f} - Test loss: {loss_test:.4f}, Test acc: {accuracy_test:.2f}')


# ### Evaluation

# In[10]:


#Plotting the metrics for Train and Test sets
orig = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [15, 5]
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Loss")
ax1.plot(train_losses, label = f"Train Loss")
ax1.plot(test_losses, label = f"Validation Loss")
ax1.legend()
ax2.set_title("Accuracy")
ax2.plot(train_accuracies, label = f"Train Acc")
ax2.plot(test_accuracies, label = f"Validation Acc")
ax2.legend()
plt.show()
plt.rcParams['figure.figsize'] = orig

#Plotting the confusion matrix
orig = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [10, 5]
ConfusionMatrixDisplay(confusion_matrix(y_test, y_hat_test), display_labels = ClassNames).plot()
plt.show()
plt.rcParams['figure.figsize'] = orig


# ## Hyperparameters 3

# ### Neural Network Model

# In[11]:


nonlinearities = ["relu", "relu"] + ["softmax"] #Last is softmax for multiclass classification
widths = [100, 200, K]
model = NeuralNet(nonlinearities, D, widths)


# ### Optimisation

# In[12]:


iterations = 300
lr = 1e-7
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []


# In[13]:


for iteration in range(iterations):
    model.train()
    
    y_prob = model.predict(x_train)
    y_hat = np.argmax(y_prob, axis = 1)
    actual_prob = y_probs(y_train, K)
    
    loss_train = loss(actual_prob, y_prob)
    accuracy_train = np.mean(y_hat == y_train)
    
    model.backprop(y_train, y_prob, lr)
    model.evaluate()
    
    y_prob_test = model.predict(x_test)
    y_hat_test = np.argmax(y_prob_test, axis = 1)
    actual_test_prob = y_probs(y_test, K)
    
    loss_test = loss(actual_test_prob, y_prob_test)
    accuracy_test = np.mean(y_hat_test == y_test)
    
    train_losses.append(loss_train)
    train_accuracies.append(accuracy_train)
    
    test_losses.append(loss_test)
    test_accuracies.append(accuracy_test)
    
    if (iteration + 1) % 1 == 0 or iteration == iterations - 1 or iteration == 0:
        print(f'Iteration {iteration + 1}/{iterations}. Train loss: {loss_train:.4f}, Train acc: {accuracy_train:.2f} - Test loss: {loss_test:.4f}, Test acc: {accuracy_test:.2f}')


# ### Evaluation

# In[15]:


#Plotting the metrics for Train and Test sets
orig = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [15, 5]
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Loss")
ax1.plot(train_losses, label = f"Train Loss")
ax1.plot(test_losses, label = f"Validation Loss")
ax1.legend()
ax2.set_title("Accuracy")
ax2.plot(train_accuracies, label = f"Train Acc")
ax2.plot(test_accuracies, label = f"Validation Acc")
ax2.legend()
plt.show()
plt.rcParams['figure.figsize'] = orig

#Plotting the confusion matrix
orig = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [10, 5]
ConfusionMatrixDisplay(confusion_matrix(y_test, y_hat_test), display_labels = ClassNames).plot()
plt.show()
plt.rcParams['figure.figsize'] = orig

