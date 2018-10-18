import torch

from Constants import WORD_EMBEDDING_LENGTH
from InformationExtraction import InformationExtraction

# Top level model for the entire network -- composed of pretrained word, event, and prediction networks
class Model(torch.nn.Module):

    def __init__(self, wordNetwork, eventNetwork, predictionNetwork):

        # Call base
        super(Model, self).__init__()

        # Add the two networks
        self.wordNetwork = wordNetwork
        self.eventNetwork = eventNetwork
        self.predictionNetwork = predictionNetwork

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

        # Create extraction object
        extractor = InformationExtraction()

        # Go through all the headlines and create event embeddings
        for headline in headlines:
            # Turn the words into structured event tuple
            structuredEvent = extractor.createStructuredTuple(headline)

            # Turn the event tuple into word embeddings -- outputs the (o1, p, o2) tuple of word embeddings
            wordEmbeddings = (self._createWordEmbeddingsForSentence(structuredEvent[0]),
                              self._createWordEmbeddingsForSentence(structuredEvent[1]),
                              self._createWordEmbeddingsForSentence(structuredEvent[2]))

            # Now have the word embeddings -- convert to the event
            eventEmbedding = self.eventNetwork(wordEmbeddings)
            eventEmbeddings = torch.cat((eventEmbeddings, eventEmbedding), 1)

        return eventEmbeddings

    # Creates a word embedding for each word in a sentence using the models word2vec network -- Averages the results for each word
    def _createWordEmbeddingsForSentence(self, sentence):

        # Separate the words
        words = sentence.split()

        # Create a word embedding for each word -- add to larger tensor
        embeddingList = torch.empty(0, 0, 0)

        for word in words:
            # Create a word embedding
            wordEmbedding = self.wordNetwork(word)

            # Add to the averaging tensor
            embeddingList = torch.cat((embeddingList, wordEmbedding), 1)

        # Average the embedding set in the 1st dimension -- reshape to have 3-d tensor
        return embeddingList.mean(1).view(1, 1, WORD_EMBEDDING_LENGTH)

# Main to sanity test this model
if __name__ == '__main__':

    w = lambda word: torch.randn(1, 1, WORD_EMBEDDING_LENGTH)

    m = Model(w, None, None)
    m((["Nvidia is sues tech Microsoft"], [], []))