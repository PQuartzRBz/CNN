import torch.nn as nn
import torch.nn.functional as F

modelA = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Dropout(p=0.2),
    nn.Flatten(),
    nn.Linear(86528 , 128),
    nn.Dropout(p=0.2),

    nn.Linear(128, 15),
    nn.Softmax(0)
)