import torch
from Constants import WORD_EMBEDDING_LENGTH
from ConfigManager import ConfigManager

# Import the Model to return and the embeddings
from Model import Model
from WordEmbedding import WordEmbedding
from EventEmbedding import EventEmbedding
from DeepPredictionNetwork import DeepPredictionNetwork

def TrainNetworksAndCreateModel(configManager):

    # Create each network
    wordNetwork = WordEmbedding(configManager=configManager, fromDatabase=False).to(configManager.device)
    eventNetwork = EventEmbedding(WORD_EMBEDDING_LENGTH).to(configManager.device)
    predictionNetwork = DeepPredictionNetwork().to(configManager.device)

    # Train each of the networks and return the full model -- Train each network separately
    for network in [wordNetwork, eventNetwork, predictionNetwork]:
        network.train()
        network.trainNetwork()
        network.eval()

    # Create the network model and return
    return Model(wordNetwork=wordNetwork, eventNetwork=eventNetwork, predictionNetwork=predictionNetwork).to(configManager.device)