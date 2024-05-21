import torch
import torch.nn
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        '''
            1. Reshape an M * N tensor into a (M * N // 2) * 2 tensor.
        '''
        # torch.reshape() will be useful - check out the documentation
        M, N = to_reshape.size()
        return torch.reshape(to_reshape, (-1, 2))

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        '''
            2. Find the average of every column in a tensor.
        '''
        # torch.mean() will be useful - check out the documentation
        return torch.mean(to_avg, 0)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        '''
            3. Combine an M * N tensor and a M * M tensor into a M * (M + N) tensor.
        '''
        # torch.cat() will be useful - check out the documentation
        return torch.cat((cat_one, cat_two), dim=1)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        '''
            4. Calculate the mean squared error loss between a prediction and target tensor.
        '''
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        return torch.nn.functional.mse_loss(prediction, target)
