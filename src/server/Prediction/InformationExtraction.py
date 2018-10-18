import spacy

class InformationExtraction():

    def __init__(self):

        # Create the nlp function
        self.nlp = spacy.load('en')

    def createStructuredTuple(self, text):

        # Call the NLP function on the text
        doc = self.nlp(text)

        # Return the tuple as desired
        print(doc)

# Main routine to test the extraction functionality
if __name__ == '__main__':
    ie = InformationExtraction()
    ie.createStructuredTuple('Nvidia sues Microsoft')