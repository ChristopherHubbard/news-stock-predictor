import torch
import pickle
import os
from datetime import date, timedelta, datetime

# Imports from user files
from InformationExtraction import InformationExtraction
from ConfigManager import ConfigManager
from Constants import WORD_EMBEDDING_LENGTH, SLICE_SIZE, MT_DAYS, MODEL_FILE_PATH
from WordEmbedding import WordEmbedding
from EventEmbedding import EventEmbedding
from DeepPredictionNetwork import DeepPredictionNetwork

# Top level model that runs the entire network
class Model(torch.nn.Module):

    def __init__(self, configManager):

        # Call base
        super(Model, self).__init__()

        # Set the network as untrained
        self.trained = False

        # Add the two networks
        self.wordNetwork = WordEmbedding(configManager)
        self.eventNetwork = EventEmbedding(WORD_EMBEDDING_LENGTH)
        self.predictionNetwork = DeepPredictionNetwork()

        # Create the information extraction utility
        try:
            self.extractor = InformationExtraction(configManager)
        except Exception as ex:
            print('Extractor failed to initialize, is the Stanford server running?')
            raise Exception('Extractor failure')

    # This is the routine that runs the entire prediction from headline -> structured tuple (o1, p, o2) -> wordEmbeddings for (o1, p, o2) -> eventEmbeddings -> prediction result
    #headlines = [{ timestamp:, headline: }]
    def forward(self, headlinesWithTime):

        # Modify to accept headlines without LT, MT, ST format, just a list of timestamped headlines

        # Get the event embeddings [{ timestamp:, embedding: }]
        eventEmbeddings = self._createEventEmbeddings(headlinesWithTime=headlinesWithTime)

        # Now order them into LT, MT, ST
        # Set up the time periods
        startPeriod = min(eventEmbeddings.keys())
        endPeriod = max(eventEmbeddings.keys())
        ltPeriod = endPeriod - startPeriod
        mtPeriod = ltPeriod.days + 1 - MT_DAYS

        # Set up the three embedding ranges
        LTEmbeddings = torch.empty(0, 0, 0)
        MTEmbeddings = torch.empty(0, 0, 0)
        for i in range(ltPeriod.days + 1):
            LTEmbeddings = torch.cat((LTEmbeddings, eventEmbeddings[startPeriod + timedelta(i)]), 1)

            # Add to MT if possible
            if i >= mtPeriod:
                MTEmbeddings = torch.cat((MTEmbeddings, eventEmbeddings[startPeriod + timedelta(i)]), 1)

        STEmbeddings = eventEmbeddings[endPeriod]

        # Now have the event embeddings -- Should be able to send to the prediction network
        return self.predictionNetwork((LTEmbeddings, MTEmbeddings, STEmbeddings))

    # Create the event embeddings for all the headlines passed in -- runs the word embedding and event embedding networks
    def _createEventEmbeddings(self, headlinesWithTime):

        # Create empty dictionary to add the embeddings and timestamps
        eventEmbeddings = {}

        # Go through all the headlines and create event embeddings
        for headlineWithTime in headlinesWithTime:
            # Turn the words into structured event tuple -- might not be possible
            try:

                # Extract the headline
                headline = headlineWithTime['headline']

                # Retrieve the embeddings for this timestamp from the dictionary if it exists
                if headlineWithTime['timestamp'] in eventEmbeddings:
                    embeddingList = eventEmbeddings[headlineWithTime['timestamp']]
                else:
                    embeddingList = torch.empty(0, 0, 0)

                # Create the word embedding for this sentence
                wordEmbeddings = self._createWordEmbeddings(headline)

                # Now have the word embeddings -- convert to the event
                eventEmbedding = self.eventNetwork(wordEmbeddings)

                # Add to the averaging tensor
                embeddingList = torch.cat((embeddingList, eventEmbedding), 1)

                # Set the list to this timestamp
                eventEmbeddings[headlineWithTime['timestamp']] = embeddingList
            except Exception as ex:
                print(ex)

        # Go through the items and average
        for key, val in eventEmbeddings.items():
            eventEmbeddings[key] = val.mean(1).view(1, 1, SLICE_SIZE)

        # Now every item is averaged -- need to replace the missing entries for the 30 days with torch.zeros
        start = min(eventEmbeddings.keys())
        end = max(eventEmbeddings.keys())
        timePeriod = end - start

        for i in range(timePeriod.days + 1):

            # Add the entry if it doesnt exist
            if (start + timedelta(i)) not in eventEmbeddings:
                eventEmbeddings[start + timedelta(i)] = torch.zeros(1, 1, SLICE_SIZE)

        # Now have all the appropriate number of days -- should be 30? -- Return the event embeddings { timestamp: , embedding: }
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
        for i in range(0, len(eventTrainingData[0]), 29):
        #for wordEmbedding, _ in eventTrainingData:

            # LT Embeddings -- 30 Embeddings concatenated into one 1, 30, Slice tensor

            # Set the input of the training data
            eventEmbedding = self.eventNetwork(wordEmbedding)

            # Set the prediction data
            # Using random value for target in this way is unsupervised (desired or not?) -- NOT
            predictionTrainingData.append((eventEmbeddings, torch.randn(1, 1, 1)))

        # Run the prediction data through the prediction network's training method
        self.predictionNetwork.trainNetwork(trainingData=predictionTrainingData)

        # Now the network should be completely trained
        self.trained = True

        # Now save the trained network (And its subnetworks to be safe)
        self._saveNetwork()

    def _accuracy(self, predicted, expected):

        # Measure the accuracy of the network
        total = len(expected)
        correct = (predicted == expected).sum()
        self.accuracy = 100 * correct / total
        print(self.accuracy)

    def _saveNetwork(self):

        # Save the pytorch models
        torch.save(self.state_dict(), MODEL_FILE_PATH)

    def _loadNetwork(self):

        # Load the pytorch models and place in eval mode
        self.load_state_dict(torch.load(MODEL_FILE_PATH))
        self.eval()

# Main to sanity test this model
if __name__ == '__main__':

    m = Model(ConfigManager('LOCAL'))
    m._loadNetwork()
    print('Did it work: YES')
    m([{
        'timestamp': datetime.strptime('30 Oct 2018', '%d %b %Y').date(),
        'headline': 'Nvidia sues Google'
    }, {
        'timestamp': datetime.strptime('29 Oct 2018', '%d %b %Y').date(),
        'headline': 'Apple sues Google'
    }, {
        'timestamp': datetime.strptime('1 Oct 2018', '%d %b %Y').date(),
        'headline': 'Google sues Apple'
    }, {
        'timestamp': datetime.strptime('16 Oct 2018', '%d %b %Y').date(),
        'headline': 'Chris is the best'
    }])