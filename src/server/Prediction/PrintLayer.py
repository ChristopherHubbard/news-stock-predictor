import torch

# Layer to print the intermediate Tensor in hidden layersz -- used as a helper
class PrintLayer(torch.nn.Module):

    def __init__(self, printSize=False):

        # Call the base constructor
        super(PrintLayer, self).__init__()

        # Setting to print the size or not
        self.printSize = printSize

    def forward(self, tensor):

        # Print the tensor passed in and return the same tensor to the next layer
        print(tensor)

        # Print size if desired
        if self.printSize:
            print(tensor.size)

        # Pass original tensor on to the next layer
        return tensor