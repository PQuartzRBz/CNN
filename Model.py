import argparse
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tools.utils import AverageMeterSet
'''
class Model



void main(){
    ..get parameter
    ..config device
    ..define model
    ..config hyperparameter
    construct model structure

    load train/test data
    train model
    test model
    
    write log(h-param, layer, accuracy)
}
'''
class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.m3 = nn.Dropout(p=0.2)

        self.d1 = nn.Linear(86528, 128)
        self.m4 = nn.Dropout(p=0.2)

        self.d2 = nn.Linear(128, 15)



    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)


        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)


        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.m3(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)
        x = self.m4(x)

        # logits => 32x10
        logits = self.d2(x)
        # x = F.softmax(logits, dim=1)
        return logits


def get_accuracy(logit, dir_test, batch_size):
        ''' Obtain accuracy for training round '''
        corrects = (torch.max(logit, 1)[1].view(dir_test.size()).data == dir_test.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()


def main(args):
    print('Start Model.main')
    writer = SummaryWriter()
    meters = AverageMeterSet()
    meters.reset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    img_width,img_height = 224,224

    path_train = os.path.join(
                args.data_path,
                'train'
                )

    path_test = os.path.join(
                args.data_path,
                'test'
                )

    transform = transforms.Compose([
            transforms.Resize(
                (img_height,
                img_width)
                ),
            transforms.ToTensor()]
            )

    trainset = torchvision.datasets.ImageFolder(
        root=path_train,
        transform=transform
        )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
        )
    
    testset = torchvision.datasets.ImageFolder(
        root=path_test,
        transform=transform
        )

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0
        )

    iter_all = 0
    train_acc = 0
    for epoch in range(1, args.epochs + 1):
        model = model.train()
        for i, (images, labels) in enumerate(trainloader):
            print('here')
            images = images.to(device)
            labels = labels.to(device)

            ## forward + backprop + loss
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()

            loss_value = loss.detach().item()
            

            writer.add_scalar('loss',loss_value,iter_all+i+1)
            meters.update(str(epoch), loss_value, n=1)
            train_acc += get_accuracy(logits, labels, args.batch_size)
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
            %(epoch, writer[str(epoch)].avg, train_acc/(i+1)))
        iter_all += i+1
    writer.close()

    test_acc = 0.0
    model.eval()
    for i, (images, labels) in enumerate(testloader, 0):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_acc += get_accuracy(outputs, labels, args.test_batch_size)
    print('Test Accuracy: %.2f'%( test_acc/(i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch MNIST Example'
        )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        metavar='N',
        help='input batch size for training (default: 32)'
        )

    parser.add_argument(
        '--test-batch-size', 
        type=int, default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)'
        )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)'
        )

    parser.add_argument(
        '--lr', 
        type=float, 
        default=1.0, 
        metavar='LR',
        help='learning rate (default: 1.0)'
        )

    parser.add_argument(
        '--data-path',
        type=str, 
        default='dataset\model1',
        metavar='N',
        help='path to data folder'
        )

    parser.add_argument(
        '--save-model', 
        action='store_true', 
        default=False,
        help='For Saving the current Model'
        )

    args = parser.parse_args()
    main(args)