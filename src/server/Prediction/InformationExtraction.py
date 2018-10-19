from pycorenlp import StanfordCoreNLP
from ConfigManager import ConfigManager

class InformationExtraction():

    def __init__(self, configManager):

        # Create the NLP using the server address for the stanford NLP -- Make sure that the server is started first
        try:
            self.nlp = StanfordCoreNLP(configManager.config['StanfordCoreNLPServer'])
        except:
            raise ConnectionError('There was an error connecting to the OpenIE engine. Make sure the server is running on the correct port')


    def createStructuredTuple(self, text):

        # Extract the tuple -- If the tuple cant be created then simply move to the next headline
        try:
            # Call the NLP function on the text -- Makes an HTTP request to the stanford NLP server
            annotation = self.nlp.annotate(text=text, properties={
                                                    'annotators': 'tokenize, ssplit, pos, depparse, natlog, openie',
                                                    'outputFormat': 'json',
                                                    'openie.triple.strict': 'true'
                                                    })

            # Retrieve the openIE result from the annotation
            result = [annotation["sentences"][0]["openie"] for item in annotation]
            openIE = result[0][0]

            # Construct the tuple
            return (openIE['subject'], openIE['relation'], openIE['object'])
        except:
            raise ValueError('There was an error creating the tuple.')

# Main routine to test the extraction functionality
if __name__ == '__main__':
    ie = InformationExtraction(ConfigManager('LOCAL'))
    print(ie.createStructuredTuple('Nvidia fourth quarter results miss views.'))
    print(ie.createStructuredTuple("Amazon eliminates worker bonuses right before the holidays."))
    print(ie.createStructuredTuple("Delta profits didn't reach goals."))
    print(ie.createStructuredTuple('Microsoft co-founder dies at age 65.'))
    print(ie.createStructuredTuple('BCG Boosts Its Expanding Private Equity Practice with Seasoned PE Value Creation Executive'))