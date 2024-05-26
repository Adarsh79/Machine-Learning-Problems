from typing import List
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # First let's get the total set of words
        words = set()
        combined = positive + negative
        for sentence in combined:
            for word in sentence.split():
                words.add(word)

        # Now let's build a mapping
        sorted_list = sorted(list(words))
        word_to_int = {}
        for i, c in enumerate(sorted_list):
            word_to_int[c] = i + 1

        # Write encode() which is used to build the dataset
        def encode(sentence):
            integers = []
            for word in sentence.split():
                integers.append(word_to_int[word])
            return integers
        
        var_len_tensors = []
        for sentence in combined:
            var_len_tensors.append(torch.tensor(encode(sentence)))
        
        return nn.utils.rnn.pad_sequence(var_len_tensors, batch_first = True)
