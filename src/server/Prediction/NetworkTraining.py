import torch
import json
from Constants import WORD_EMBEDDING_LENGTH
from ConfigManager import ConfigManager

# Import the Model to return and the embeddings
from InformationExtraction import InformationExtraction
from Model import Model
from WordEmbedding import WordEmbedding
from EventEmbedding import EventEmbedding
from DeepPredictionNetwork import DeepPredictionNetwork

def TrainNetworksAndCreateModel(configManager):

    # Create each network
    wordNetwork = WordEmbedding(configManager=configManager, fromDatabase=False).to(configManager.device)

    # Load the headlines and input them into the word network
    headlines = []
    for file in json_files:
        data = json.loads(file)
        headlines += data['Headlines']

    # Train the event network
    eventNetwork = EventEmbedding(WORD_EMBEDDING_LENGTH, eventTrainingData).to(configManager.device)

    predictionNetwork = DeepPredictionNetwork(predictionTrainingData).to(configManager.device)

    # Train each of the networks and return the full model -- Train each network separately
    for network in [wordNetwork, eventNetwork, predictionNetwork]:
        network.train()
        network.trainNetwork()
        network.eval()

    # Create the network model and return
    return Model(wordNetwork=wordNetwork, eventNetwork=eventNetwork, predictionNetwork=predictionNetwork).to(configManager.device)