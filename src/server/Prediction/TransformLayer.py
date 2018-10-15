import torch

# Layer to transform a tensor in a sequential NN to a different format -- useful to define networks to output correctly shaped tensors
# Also helps include intermediate transformations between layers
class TransformLayer(torch.nn.Module):

    def __init__(self, toSize):

        # Call the base constructor
        super(TransformLayer, self).__init__()

        self.toSize = toSize

    def forward(self, tensor):

        # Transform the tensor into the appropriate shape -- this could throw an exception, but that should be the desired behavior
        batch_size, rows, columns = self.toSize
        return tensor.view(batch_size, rows, columns)
