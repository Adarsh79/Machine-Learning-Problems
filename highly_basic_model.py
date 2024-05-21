import torch
import torch.nn as nn

'''
Implement a neural network with the following specifications
Input layer has 4 features
2 hidden layers each having 6 nodes
Output layer having 2 nodes
'''

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(4, 6)
        self.second_layer = nn.Linear(6, 6)
        self.final_layer = nn.Linear(6, 2)
    
    def forward(self, X):
        return self.final_layer(self.second_layer(self.first_layer.forward(X)))