import torch
from typing import List, Tuple

class Solution:
    '''
        Before we can train a transformer like GPT, we need to define the dataset. 
        We take a giant body of text and we can create examples for the model to predict the next token based on different contexts. 
        This is what “ChatGPT was trained on the entire internet” means.

        Your task is to write the `batch_loader()` function which will generate a `batch_size * context_length` dataset and its labels. 
        Use `torch.randint()` to pick batch_size different starting words for each sequence.
    '''
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        words = raw_dataset.split()
        indices = torch.randint(low=0, high=len(words)-context_length, size=(batch_size, 1))
        X = []
        Y = []
        for idx in indices:
            X.append(words[idx:idx+context_length])
            Y.append(words[idx+1:idx+context_length+1])
        
        return X, Y
