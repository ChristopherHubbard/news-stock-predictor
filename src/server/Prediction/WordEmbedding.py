import torch
import numpy as np

from Constants import WORD_EMBEDDING_LENGTH
from ConfigManager import ConfigManager

# Word Embedding layer -- Used with pretrained word embeddings from GloVe
class WordEmbedding(torch.nn.Module):

    def __init__(self, configManager, fromDatabase=True):

        # Call the base constructor
        super(WordEmbedding, self).__init__()

        self.trained = True

        # Set up the network that converts strings to word embeddings -- Create the vocab if needed
        # Load the word embeddings from the database
        if fromDatabase:
            self._vocabDict = {}

            # How to hit the database? Add a data layer and then request for the words in the sentence?
        else:
            # Initialize empty dictionary for the vocabulary
            self._vocabDict = {}
            with open(f'{configManager.config["GloVePath"]}/glove.6B.{WORD_EMBEDDING_LENGTH}d.txt', 'rb') as f:

                # Go through the lines in the file
                for l in f:

                    # Decode and split line -- then retrieve word
                    line = l.decode().split()
                    word = line[0]

                    # Create the PyTorch Tensor
                    embedData = [[list(map(float, line[1:]))]]
                    embedding = torch.tensor(data=embedData, dtype=torch.double, device=configManager.device)
                    self._vocabDict[word] = embedding


    def forward(self, word):

        # Create the word embedding by calling the network functions -- GloVe should be loaded
        # Use lowercase of string for the pre-trained dictionary
        return self._vocabDict[word.lower()]

    # This network is pretrained -- only have this to maintain the nn interface
    def trainNetwork(self):

        self.trained = True
        # Return self (even though no training)
        return self

# Main function for initial testing
if __name__ == '__main__':
    w = WordEmbedding(configManager=ConfigManager('LOCAL'), fromDatabase=False)
    word = '-'
    print(w)
    print(w.forward(word))
