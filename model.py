import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 16, 3, padding = 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   
                                   nn.Conv2d(16, 32, 3, padding = 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Dropout2d(dropout),
                                   
                                   nn.Conv2d(32, 64, 3, padding = 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   
                                   nn.Conv2d(64, 128, 3, padding = 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   
                                   nn.Conv2d(128, 192, 3, padding = 1),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   
                                   nn.Conv2d(192, 256, 3, padding = 1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                
                                   nn.Flatten(),
                                   nn.Linear(2304, 1000),
                                   nn.BatchNorm1d(1000),
                                   nn.ReLU(),
                                   
                                   nn.Linear(1000, 500),
                                   nn.BatchNorm1d(500),
                                   nn.ReLU(),
                                   
                                   nn.Linear(500, 250),
                                   nn.BatchNorm1d(250),
                                   nn.ReLU(),
                                   
                 
                                   
                                   nn.Linear(250,num_classes))
                                   
                                   
        

      
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
