# Imports
import torch
from Constants import WORD_EMBEDDING_LENGTH, SLICE_SIZE

# No need to import numpy since pytorch uses tensors instead of direct np arrays

# Create a Neural Tensor Network -- Used to determine the relationship between two vectors
# which in this case are the word embedding vectors for the actor, action, and object
# according to the pyramid-esque diagram

class EventEmbedding(torch.nn.Module):

    # Initialize the Neural Tensor Layer -- takes in the size of each input to the network
    def __init__(self, embeddingLength):

        # Call the base constructor
        super(EventEmbedding, self).__init__()

        # Set the input sizes
        self.actor_size = embeddingLength
        self.action_size = embeddingLength
        self.object_size = embeddingLength

        # Set the x and y dimensions of the tensors -- both are d
        self.d = WORD_EMBEDDING_LENGTH

        # Set the slice size of the tensor -- what should this be set to??
        self.k = SLICE_SIZE

        # Define the sequence of layers for the whole network -----------------------------------------------------------------

        # BiLinear layer for xAy + b -- A is weight (Tensor?) -- includes the bias parameter b -- Make sure biLinear 3 and linear 3 work for these sizes
        self.biLinear1 = torch.nn.Bilinear(in1_features=self.actor_size, in2_features=self.action_size, out_features=self.k, bias=True)
        self.biLinear2 = torch.nn.Bilinear(in1_features=self.action_size, in2_features=self.object_size, out_features=self.k, bias=True)
        self.biLinear3 = torch.nn.Bilinear(in1_features=self.k, in2_features=self.k, out_features=self.k, bias=True)

        # Add the W[x, y] to the output of the layer?
        self.linear1 = torch.nn.Linear(in_features=(self.actor_size + self.action_size), out_features=self.k, bias=True)
        self.linear2 = torch.nn.Linear(in_features=(self.action_size + self.object_size), out_features=self.k, bias=True)
        self.linear3 = torch.nn.Linear(in_features=(self.k + self.k), out_features=self.k, bias=True)

        # Activation function for the result
        self.activation = torch.nn.Tanh()

    # Result of a forward pass -- This should return the event embeddings
    def forward(self, event):
        # Extract the actor, action, and object
        o1, p, o2 = event

        # Transpose o1 and o2 for first bilinear transform
        o1_T = torch.transpose(o1, 1, 2)
        o2_T = torch.transpose(o2, 1, 2)

        stacked_o1_P = torch.cat((o1, p), -1)
        stacked_o2_P = torch.cat((p, o2), -1)

        r1 = self.biLinear1(o1, p) + self.linear1(stacked_o1_P)
        r2 = self.biLinear2(p, o2) + self.linear2(stacked_o2_P)

        # Final run with r1 and r2 -- transpose
        r1_T = torch.transpose(r1, 1, 2)
        r2_T = torch.transpose(r2, 1, 2)

        # How to insure that these concats yield the appropriate results?
        stacked_r1_r2 = torch.cat((r1, r2), -1)

        u = self.biLinear3(r1, r2) + self.linear3(stacked_r1_r2)

        # Run the activation function with the result
        return self.activation(u)

    # Method to train the event embedding network -- calculate loss and use standard backpropagation
    def trainNetwork(self):

        # Return the network after training
        return self

# Testing for the Event embedding NTN -- Numbers are weird on initial test but training will probably produce correct result
if __name__ == '__main__':

    o1 = torch.randn(1, 1, WORD_EMBEDDING_LENGTH)
    p = torch.randn(1, 1, WORD_EMBEDDING_LENGTH)
    o2 = torch.randn(1, 1, WORD_EMBEDDING_LENGTH)
    wordEmbeddings = o1, p, o2
    event = EventEmbedding(WORD_EMBEDDING_LENGTH)
    print(event(wordEmbeddings))