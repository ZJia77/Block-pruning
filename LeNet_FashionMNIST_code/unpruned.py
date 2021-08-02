import torch as pt
import torchvision as ptv
import numpy as np
import math
import lenet


#train the unpruned model

device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")

#FashionMNIST dataset
train_dataset = ptv.datasets.FashionMNIST("./",download=True,transform = ptv.transforms.Compose([ptv.transforms.ToTensor(),ptv.transforms.Normalize([0.5], [0.5])]))
test_dataset = ptv.datasets.FashionMNIST("./",train=False,transform = ptv.transforms.Compose([ptv.transforms.ToTensor(),ptv.transforms.Normalize([0.5], [0.5])]))
#loader
trainloader = pt.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=128)
testloader = pt.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=128)

model = lenet.net().to(device)
lossfunc = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(),5e-4)        


#test accuracy of model
def test_model(model,testloader):
    test_total = 0
    test_correct = 0    
    for data, label in testloader:
        data = data.to(device)
        label = label.to(device)
        outputs = model(data).to(device)        
        _, predicted = pt.max(outputs,dim=1)
        test_total += label.size(0)
        test_correct += ((predicted==label).sum()).item()
    print("test_acc")
    print(test_correct / test_total)
    return test_correct / test_total

#training accuracy of model
def train_model(model,trainloader):
    train_total = 0
    train_correct = 0    
    for data, label in trainloader:
        data = data.to(device)
        label = label.to(device)                
        outputs = model(data).to(device)
        _, predicted = pt.max(outputs,dim=1)
        train_total += label.size(0)
        train_correct += ((predicted==label).sum()).item()
    print("train_acc")
    print(train_correct / train_total)
    return train_correct / train_total


#training the unpruned model
for epoch in range(1000):
    for i,(data,label) in enumerate(trainloader):
        data = data.to(device)
        label = label.to(device)
        model.zero_grad()
        outputs = model(data).to(device)
        loss = lossfunc(outputs,label).to(device)
        loss.backward()
        optimizer.step()
    print(epoch)
    test_result=test_model(model,testloader)
    training_result=train_model(model,trainloader)
    pt.save(model.state_dict(),"./base_training.pt")


