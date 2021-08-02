import torch as pt
import torchvision as ptv
import resnet18
import os
import numpy as np
import math

batch_size = 1024
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")

#  data pre-treatment
data_transform = {
    "train": ptv.transforms.Compose([ptv.transforms.RandomCrop(32, padding=4),
                                 ptv.transforms.RandomHorizontalFlip(),
                                 ptv.transforms.ToTensor(),
                                 ptv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    "val": ptv.transforms.Compose([
                               ptv.transforms.ToTensor(),
                               ptv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])}

# load train data
trainset = ptv.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=data_transform["train"])

trainloader = pt.utils.data.DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

# load test data
testset = ptv.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=data_transform["val"])
testloader = pt.utils.data.DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)

model = resnet18.ResNet18()
'''
for i in model.state_dict():

    print(i)
'''

model.to(device)  # net into cuda
# change fc layer structure
inchannel = model.linear.in_features

print(inchannel)
# class number in Cifar10
class_num = 10


# add a 300-neurons fully-connected layers

model.linear = pt.nn.Sequential(
        pt.nn.Linear(inchannel, 300),
        pt.nn.ReLU(),
        pt.nn.Linear(300, class_num),        
        pt.nn.Softmax(dim=1)
    )

model.to(device)

'''
for i in model.state_dict():

    print(i)

model_fc1_tensor1=model.state_dict()["linear.0.weight"].float()
print(model_fc1_tensor1.size(),type(model_fc1_tensor1))
model_fc1_tensor1=model.state_dict()["linear.0.bias"].float()
print(model_fc1_tensor1.size(),type(model_fc1_tensor1))
model_fc1_tensor1=model.state_dict()["linear.2.weight"].float()
print(model_fc1_tensor1.size(),type(model_fc1_tensor1))
model_fc1_tensor1=model.state_dict()["linear.2.bias"].float()
print(model_fc1_tensor1.size(),type(model_fc1_tensor1))
'''



for param in model.parameters():
    param.requires_grad = True




# define optimizer and loss function
lossfunc = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)



best_acc=0.0
epochs=300


start_epoch = 1
#unpruned
for epoch in range(start_epoch, epochs + 1):
    #training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for i, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(data).to(device)
        loss = lossfunc(outputs, label).to(device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += label.size(0)
        train_correct += predicted.eq(label).sum().item()
    print('TrainLoss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (i + 1), 100. * train_correct / train_total, train_correct, train_total))
    print(epoch)
    
    #test
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    test_acc = 0.0  
    with pt.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).to(device)
            loss = lossfunc(outputs, targets).to(device)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'TestLoss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * test_correct / test_total, test_correct, test_total))

    acc = test_correct / test_total
    if acc > best_acc:
        best_acc = acc

        pt.save(model.state_dict(),"./result_pt/epoch.pt")
        

