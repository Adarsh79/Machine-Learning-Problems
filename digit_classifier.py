import torch
import torch.nn as nn
from torchtyping import TensorType
import matplotlib.pyplot as plt

class Solution(nn.Module):
    '''
        Task: Implement a neural network that can recognize black and white images of handwritten digits.
            This is simple but powerful application of neural networks.
        Architecture: Use a linear layer with 512 neurons followed by a ReLU activation, as well as a dropout layer with probability p = 0.2 that precedes a final Linear layer with 10 neurons and a sigmoid activation.
            Each output neuron corresponds to a digit from 0 to 9, where each value is the probability that the input image is the corresponding digit.
    '''
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        self.first_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.final_layer = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        '''
            input - one or more 28 * 28 black and white images of handwritten digits. `len(images) > 0` and `len(images[i]) = 28 * 28` for `0 <= i < len(images)`.
        '''
        torch.manual_seed(0)
        # Return the model's prediction in 4 decimal places
        out = torch.round(self.sigmoid(self.final_layer(self.dropout(self.relu(self.first_layer(images))))), decimals=4)
        return out


model = Solution()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 5
for epoch in range(epochs):
    for images, labels in train_dataloader:
        images = images.view(images.shape[0], 784)

        # Training body
        model_prediction = model(images)
        optimizer.zero_grad()
        loss = loss_function(model_prediction, labels)
        loss.backward() # To calculate the derivates of all the weights
        optimizer.step() # new_w = old_w - derivative * learning_rate

model.eval()
for images, labels in test_dataloader:
    images = images.view(images.shape[0], 784)

    model_prediction = model(images)
    max, idx = torch.max(model_prediction, dim=1)
    for i in range(len(images)):
        plt.imshow(images[i].view(28, 28))
        plt.show()
        print(idx[i].item())
    break