import torch
from Constants import WORD_EMBEDDING_LENGTH

# Import the Model to return and the embeddings
from Model import Model
from WordEmbedding import WordEmbedding
from EventEmbedding import EventEmbedding
from DeepPredictionNetwork import DeepPredictionNetwork

def TrainNetworksAndCreateModel():

    # Create each network
    wordNetwork = WordEmbedding()
    eventNetwork = EventEmbedding(WORD_EMBEDDING_LENGTH)
    predictionNetwork = DeepPredictionNetwork()

    # Train each of the networks and return the full model -- Train each network separately
    for network in [wordNetwork, eventNetwork, predictionNetwork]:
        network.trainNetwork()

    # Create the network model and return
    return Model(wordNetwork=wordNetwork, eventNetwork=eventNetwork, predictionNetwork=predictionNetwork)