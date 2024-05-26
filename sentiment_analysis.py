# Sentiment Analysis

'''
Let's use the sentiment analysis model you created to train a model using a dataset of Amazon product reviews, where we have the corresponding sentiment as a number between -1 and 1 (completely negative to completely positive).

Note: Before running anything, go to Runtime -> Change Runtime Type -> T4 GPU to speed things up.

First, we'll download the raw text file.
'''

# !git clone https://github.com/gptandchill/sentiment-analysis
# %cd sentiment-analysis

'''
Now that we have the dataset, let's convert it into PyTorch tensors. You can just run and ignore the details of parsing the text file.
'''

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

list_of_strings = []
list_of_labels = []

import csv
with open('EcoPreprocessed.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      list_of_strings.append(row[1])
      list_of_labels.append(float(row[2]))


'''
To create the actual tensors, we can borrow the solution code from the "NLP Intro" problem.
'''

def get_dataset(list_of_strings):

    # First let's get the total set of words
    words = set()
    for sentence in list_of_strings:
        for word in sentence.split():
            words.add(word)

    vocab_size = len(words)

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
    for sentence in list_of_strings:
        var_len_tensors.append(torch.tensor(encode(sentence)))

    return vocab_size + 1, nn.utils.rnn.pad_sequence(var_len_tensors, batch_first = True), word_to_int

vocab_size, training_dataset, word_to_int = get_dataset(list_of_strings)
training_labels = torch.unsqueeze(torch.tensor(list_of_labels), dim = -1)

'''
Now that the dataset is in, let's use the model you wrote. The only change we will make is replacing the Sigmoid layer with a Tanh layer. Tanh outputs are always between -1 and 1, so that makes more sense given the data labels.
'''

class EmotionPredictor(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dimension: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dimension)
        self.linear_layer = nn.Linear(embedding_dimension, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        averaged = torch.mean(embeddings, axis = 1)
        projected = self.linear_layer(averaged)
        return self.tanh(projected)

'''
Below we train the model using our standard training loop. One difference you will notice is that I choose a random 64 sized subset of the total dataset (thousands and thousands of examples), which speeds up training.
'''

embedding_dimension = 256
model = EmotionPredictor(vocab_size, embedding_dimension)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for i in range(1000):
  randperm = torch.randperm(len(training_dataset))
  training_dataset, training_labels = training_dataset[randperm], training_labels[randperm]
  mini_batch = training_dataset[:64]
  mini_batch_labels = training_labels[:64]
  pred = model(mini_batch)
  optimizer.zero_grad()
  loss = loss_function(pred, mini_batch_labels)
  if i % 100 == 0:
    print(loss.item())
  loss.backward()
  optimizer.step()

'''
Now let's see the model's outputs on some examples it's never seen before!
'''



example_one = "worst movie ever"

example_two = "best movie ever"

example_three = "weird but funny movie"

examples = [example_one] + [example_two] + [example_three]

# Let's encode these strings as numbers using the dictionary from earlier
var_len = []
for example in examples:
  int_version = []
  for word in example.split():
    int_version.append(word_to_int[word])
  var_len.append(torch.tensor(int_version))

testing_tensor = torch.nn.utils.rnn.pad_sequence(var_len, batch_first=True)
model.eval()

print(model(testing_tensor).tolist())


'''
You should find that the model outputs something close to -1 for example one, something very close to 1 for example two, and something close to 0 for example three (neutral)!

This was a very simple model, and we will build more complex ones in the next problems!
'''