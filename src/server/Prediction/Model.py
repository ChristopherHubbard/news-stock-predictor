import torch
import pickle
import os

# Imports from user files
from InformationExtraction import InformationExtraction
from ConfigManager import ConfigManager
from Constants import WORD_EMBEDDING_LENGTH
from WordEmbedding import WordEmbedding

# Top level model that runs the entire network
class Model(torch.nn.Module):

    def __init__(self, wordNetwork, eventNetwork, predictionNetwork, configManager):

        # Call base
        super(Model, self).__init__()

        # Set the network as untrained
        self.trained = False

        # Add the two networks
        self.wordNetwork = wordNetwork
        self.eventNetwork = eventNetwork
        self.predictionNetwork = predictionNetwork

        # Create the information extraction utility
        try:
            self.extractor = InformationExtraction(configManager)
        except Exception as ex:
            print('Extractor failed to initialize, is the Stanford server running?')
            raise Exception('Extractor failure')

    # This is the routine that runs the entire prediction from headline -> structured tuple (o1, p, o2) -> wordEmbeddings for (o1, p, o2) -> eventEmbeddings -> prediction result
    def forward(self, headlines):

        LTHeadlines, MTHeadlines, STHeadlines = headlines

        # Get the embeddings
        LTEmbeddings = self._createEventEmbedding(LTHeadlines)
        MTEmbeddings = self._createEventEmbedding(MTHeadlines)
        STEmbeddings = self._createEventEmbedding(STHeadlines)

        # Now have the event embeddings -- Should be able to send to the prediction network
        return self.predictionNetwork((LTEmbeddings, MTEmbeddings, STEmbeddings))

    # Create the event embeddings for all the headlines passed in -- runs the word embedding and event embedding networks
    def _createEventEmbedding(self, headlines):

        # Create empty tensor to add the embeddings
        eventEmbeddings = torch.empty(0, 0, 0)

        # Go through all the headlines and create event embeddings
        for headline in headlines:
            # Turn the words into structured event tuple -- might not be possible
            try:
                # Create the word embedding for this sentence
                wordEmbeddings = self._createWordEmbeddings(headline)

                # Now have the word embeddings -- convert to the event
                eventEmbedding = self.eventNetwork(wordEmbeddings)
                eventEmbeddings = torch.cat((eventEmbeddings, eventEmbedding), 1)
            except Exception as ex:
                print(ex)
                print('The structured tuple was not able to be constructed for headline: ' + headline)

        return eventEmbeddings

    # Takes a headline and creates word embeddings after creating a structured tuple
    def _createWordEmbeddings(self, headline):

        # Create the structured tuple and word embeddings
        try:
            structuredEvent = self.extractor.createStructuredTuple(headline)

            # Turn the event tuple into word embeddings -- outputs the (o1, p, o2) tuple of word embeddings
            wordEmbeddings = (self._createWordEmbeddingsForSentence(structuredEvent[0]),
                              self._createWordEmbeddingsForSentence(structuredEvent[1]),
                              self._createWordEmbeddingsForSentence(structuredEvent[2]))
            return wordEmbeddings
        except:
            print('The structured tuple was not able to be constructed for the headline: {}'.format(headline))
            raise Exception('The tuple for this headline couldn\'t be created')

    # Creates a word embedding for each word in a sentence using the models word2vec network -- Averages the results for each word
    def _createWordEmbeddingsForSentence(self, sentence):

        # Separate the words
        words = sentence.split()

        # Create a word embedding for each word -- add to larger tensor
        embeddingList = torch.empty(0, 0, 0, dtype=torch.double)

        for word in words:
            # Create a word embedding
            wordEmbedding = self.wordNetwork(word)

            # Add to the averaging tensor
            embeddingList = torch.cat((embeddingList, wordEmbedding), 1)

        # Average the embedding set in the 1st dimension -- reshape to have 3-d tensor
        return embeddingList.mean(1).view(1, 1, WORD_EMBEDDING_LENGTH)

    # Method to train the network end to end
    def trainNetwork(self):

        # Load all the headlines into a list
        path = './data'
        headlines = []
        for f in os.listdir(path):
            file = open(path + '/' + f, 'rb')
            data = pickle.load(file, encoding='utf-8')
            headlines += map(lambda item: item['title'], data)

        # Pass all the headlines into the event network after creating corrupt headlines
        eventTrainingData = []
        for headline in headlines:

            try:

                # Create a structured tuple for this headline
                wordEmbeddings = self._createWordEmbeddings(headline)

                # Create the structured tuple and then corrupt it
                st = self.extractor.createStructuredTuple(headline)
                corruptTuple = self.extractor.createCorruptStructuredTuple(structuredTuple=st, vocabDict=self.wordNetwork._vocabDict)
                corruptEmbeddings = (self._createWordEmbeddingsForSentence(corruptTuple[0]),
                                     self._createWordEmbeddingsForSentence(corruptTuple[1]),
                                     self._createWordEmbeddingsForSentence(corruptTuple[2]))

                # Add to the event training data
                eventTrainingData.append((wordEmbeddings, corruptEmbeddings))

            # If an exception is thrown just pass on
            except:
                pass

        # Now have the event training data -- pass it to the event networks training method
        self.eventNetwork.trainNetwork(trainingData=eventTrainingData)

        # Now the event network should be trained -- pass the original training data (non corrupt) to get the input to the prediction network
        predictionTrainingData = ()
        for wordEmbedding, _ in eventTrainingData:

            # Set the input of the training data
            eventEmbedding = self.eventNetwork(wordEmbedding)

            # Set the prediction data
            predictionTrainingData[0].append(eventEmbedding)

        # Create a randomly initialized output tensor
        predictionTrainingData[1] = torch.randn(1, 1, 1)

        # Run the prediction data through the prediction network's training method
        self.predictionNetwork.trainNetwork(trainingData=predictionTrainingData)

        # Now the network should be completely trained
        self.trained = True

# Main to sanity test this model
if __name__ == '__main__':

    w = WordEmbedding(configManager=ConfigManager('LOCAL'), fromDatabase=False)
    e = lambda event: torch.randn(1, 1, 100)
    p = lambda predict: torch.randn(1, 1, 1)

    m = Model(w, e, p, ConfigManager('LOCAL'))
    m.trainNetwork()
    m((['Nvidia fourth quarter results miss views',
        'Amazon eliminates worker bonuses right before the holidays',
        'Delta profits didn\'t reach goals',
        'Chris ate green beans'], [], []))