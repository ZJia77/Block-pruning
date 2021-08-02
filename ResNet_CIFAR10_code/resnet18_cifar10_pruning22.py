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
                                        download=False,
                                        transform=data_transform["train"])

trainloader = pt.utils.data.DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

# load test data
testset = ptv.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=False,
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
# load pretrain weights

model.to(device)  # net into cuda
# change fc layer structure
inchannel = model.linear.in_features
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
print(inchannel)

# For each weight of the model,
# fix the parameters so that they do not back propagate

for param in model.parameters():
    param.requires_grad = False

# do not fix the parameters in fully-connected layers
for param in model.linear.parameters():
    param.requires_grad = True

# This function is to compute the sum of weights in each block
# after block division of the weight matrix
def division(number1, number2, model1_fc1_tensor, model1_fc2_tensor):
    tensor1 = pt.zeros(math.ceil(300 / number1), math.ceil(512 / number2))
    tensor2 = pt.zeros(math.ceil(10 / number1), math.ceil(300 / number2))
    # the number of weights of each sub-block in every layer
    sumnum1 = 0
    sumnum2 = 0
    #compute the average of absolute values of weights in sub-blocks
    #layer1
    for m in range(math.ceil(300 / number1)):
        for n in range(math.ceil(512 / number2)):
            for i in range(number1):
                for j in range(number2):
                    if number1 * m + i <= 300 - 1 and number2 * n + j <= 512 - 1:
                        tensor1[m][n] = tensor1[m][n] + pt.abs(model1_fc1_tensor[number1 * m + i][number2 * n + j])
                        sumnum1 = sumnum1 + 1
            tensor1[m][n] = tensor1[m][n] / sumnum1
            sumnum1 = 0
    max_weight1 = pt.max(abs(tensor1))
    for m in range(math.ceil(300 / number1)):
        for n in range(math.ceil(512 / number2)):
            if tensor1[m][n] != 0:
                tensor1[m][n] = tensor1[m][n] / max_weight1
    #layer2            
    for m in range(math.ceil(10 / number1)):
        for n in range(math.ceil(300 / number2)):
            for i in range(number1):
                for j in range(number2):

                    if number1 * m + i <= 10 - 1 and number2 * n + j <= 300 - 1:
                        tensor2[m][n] = tensor2[m][n] + pt.abs(model1_fc2_tensor[number1 * m + i][number2 * n + j])
                        sumnum2 = sumnum2 + 1
            tensor2[m][n] = tensor2[m][n] / sumnum2
            sumnum2 = 0
    max_weight2 = pt.max(abs(tensor2))
    for m in range(math.ceil(10 / number1)):
        for n in range(math.ceil(300 / number2)):
            if tensor2[m][n] != 0:
                tensor2[m][n] = tensor2[m][n] / max_weight2


    return tensor1, tensor2


'''
input: the output of division function
output: the pruned weight matrix
'''
# This function is used to iteratively prune the sub-block
def pruning(new_tensor1, new_tensor2):
    #sort the weight matrix in each layer
    sorted_tensor1 = np.sort(new_tensor1[new_tensor1 != 0], axis=None)
    # 0.2 means the pruning rate in this layer
    cutoff_index1 = np.round(0.2 * sorted_tensor1.size).astype(int)
    cutoff1 = sorted_tensor1[cutoff_index1]
    new_tensor1 = np.where(new_tensor1 < cutoff1, 0, new_tensor1)
    

    sorted_tensor2 = np.sort(new_tensor2[new_tensor2 != 0], axis=None)
    # 0.2 means the pruning rate in this layer
    cutoff_index2 = np.round(0.2 * sorted_tensor2.size).astype(int)
    cutoff2 = sorted_tensor2[cutoff_index2]
    new_tensor2 = np.where(new_tensor2 < cutoff2, 0, new_tensor2)
    return new_tensor1, new_tensor2


# a mask can be set based on the corresponding weights
# and make element-wise product with the weight gradient to ensure
# that the removed weight values are not updated.
def grad_puring(model):
    for i in model.linear.parameters():
        mask = i.clone()
        mask[mask != 0] = 1
        i.grad.data.mul_(mask)




#Sparsity evaluation function
def print_sparse(model):
    result = []
    total_num = 0
    total_sparse = 0
    print("-----------------------------------")
    print("Layer sparsity")

    layer1_num = model.state_dict()["linear.0.weight"].float().view(-1).shape[0]
    total_num += layer1_num
    sparse1 = pt.nonzero(model.state_dict()["linear.0.weight"]).shape[0]
    total_sparse += sparse1
    layer1_num_bias = model.state_dict()["linear.0.bias"].float().view(-1).shape[0]
    total_num += layer1_num_bias
    sparse1_bias = pt.nonzero(model.state_dict()["linear.0.bias"]).shape[0]
    total_sparse += sparse1_bias
    print("\t", "linear.0.weight", (sparse1) / layer1_num)
    print("\t", "linear.0.bias", (sparse1_bias) / layer1_num_bias)
    result.append((sparse1) / layer1_num)
    result.append((sparse1_bias) / layer1_num_bias)

    layer2_num = model.state_dict()["linear.2.weight"].float().view(-1).shape[0]
    total_num += layer2_num
    sparse2 = pt.nonzero(model.state_dict()["linear.2.weight"]).shape[0]
    total_sparse += sparse2
    layer2_num_bias = model.state_dict()["linear.2.bias"].float().view(-1).shape[0]
    total_num += layer2_num_bias
    sparse2_bias = pt.nonzero(model.state_dict()["linear.2.bias"]).shape[0]
    total_sparse += sparse2_bias
    print("\t", "linear.2.weight", (sparse2) / layer2_num)
    print("\t", "linear.2.bias", (sparse2_bias) / layer2_num_bias)
    result.append((sparse2) / layer2_num)
    result.append((sparse2_bias) / layer2_num_bias)
    total = total_sparse / total_num
    print("Total:", total)
    thesum = sparse1 + sparse2
    num = layer1_num + layer2_num
    sp = thesum / num
    print(sp)
    return total

#test accuracy
def test_model(model,testloader):
    test_total=0;
    test_correct=0
    with pt.no_grad():
        for data,label in testloader:
            model.eval()
            data = data.to(device)
            label = label.to(device)
            outputs = model(data).to(device)
            _, predicted = outputs.max(1)
            test_total += label.size(0)
            test_correct += predicted.eq(label).sum().item()
        print('TestAcc: %.3f%% (%d/%d)' % ( 100. * test_correct / test_total, test_correct, test_total))
    return test_correct / test_total


# define optimizer and loss function
lossfunc = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)


'''
input:
rounds: number of rounds of iterative block pruning
        For example, according to the pruning rate in this paper,
        the density reaches 8% at the 10th round of pruning.
        
        The number of rounds facilitates the naming of model saving files.
        
        Special definition: The round of the unpruned baseline model is defined as -1,
        which is convenient for quickly calling the corresponding saving model file
        during iterative pruning.
number1, number2: the block pruning granularity(2)
'''
# iterative pruning function
def iterative_pruning_and_training(rounds,number1,number2):
    # When the sparsity is not high, the model converges faster.
    # In order to reduce the training time, the number of training is relatively
    # less in the first few rounds of block pruning
    epoch = 300

    
    model.load_state_dict(pt.load("./22_pruning/pruning_" + str(rounds) + ".pt"))
    test_model(model, testloader)
    print_sparse(model)

    # Get the weight matrix
    model2 = model.to(device)
    model_fc1_tensor = model2.state_dict()["linear.0.weight"].float()
    print(model_fc1_tensor.size(), type(model_fc1_tensor))
    model_fc2_tensor = model2.state_dict()["linear.2.weight"].float()
    print(model_fc2_tensor.size(), type(model_fc2_tensor))
    #division
    new_tensor1, new_tensor2 = division(number1,number2, model_fc1_tensor, model_fc2_tensor)
    #iterative pruning
    new_tensor1, new_tensor2 = pruning(new_tensor1, new_tensor2)
    params = list(model2.named_parameters())
    
    for m in range(math.ceil(300 / number1)):
        for n in range(math.ceil(512 / number2)):
            if new_tensor1[m][n] == 0:
                for i in range(number1):
                    for j in range(number2):
                        if number1 * m + i <= 300 - 1 and number2 * n + j <= 512 - 1:
                            model2.state_dict()["linear.0.weight"][number1 * m + i][number2 * n + j] = 0

    #According to the pruned sub-block, set the weights corresponding to the sub-block position to zero
    for m in range(math.ceil(10 / number1)):
        for n in range(math.ceil(300 / number2)):
            if new_tensor2[m][n] == 0:
                for i in range(number1):
                    for j in range(number2):
                        if number1 * m + i <= 10 - 1 and number2 * n + j <= 300 - 1:
                            model2.state_dict()["linear.2.weight"][number1 * m + i][number2 * n + j] = 0


    print("before fine-tuning")
    test_acc=test_model(model2, testloader)
    print_sparse(model2)
    print("begin training")
        
    model2.train()
    train_correct = 0
    train_total = 0
    optimizer = pt.optim.Adam(model2.parameters(), lr=0.0001,weight_decay=5e-4)
    lossfunc = pt.nn.CrossEntropyLoss()
    for i, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model2(data).to(device)
        loss = lossfunc(outputs, label).to(device)
        loss.backward()
        grad_puring(model2)
        optimizer.step()
        _, predicted = outputs.max(1)
        train_total += label.size(0)
        train_correct += predicted.eq(label).sum().item()
    
    print('TrainAcc: %.3f%% (%d/%d)' % (
    100. * train_correct / train_total, train_correct, train_total))

    train_acc = 100. * train_correct / train_total


    
    pt.save(model2.state_dict(), "./22_pruning/pruning_" + str(rounds + 1) + ".pt")
    model2.load_state_dict(pt.load("./22_pruning/pruning_" + str(rounds + 1) + ".pt"))


    model2.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with pt.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model2(inputs).to(device)
            loss = lossfunc(outputs, targets).to(device)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'TestAcc: %.3f%% (%d/%d)'
              % ( 100. * test_correct / test_total, test_correct, test_total))
    test_acc=100. * test_correct / test_total

    train_epoch = 1
    print("Training epoch is "+str(train_epoch))
    
    
    while train_epoch < epoch:
        train_epoch = train_epoch + 1
        print("training epoch is " + str(train_epoch))
        #training
        model2.train()
        train_correct = 0
        train_total = 0
        optimizer = pt.optim.Adam(model2.parameters(), lr=0.0001, weight_decay=5e-4)
        lossfunc = pt.nn.CrossEntropyLoss()
        for i, (data, label) in enumerate(trainloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model2(data).to(device)
            loss = lossfunc(outputs, label).to(device)
            loss.backward()
            grad_puring(model2)
            optimizer.step()
            _, predicted = outputs.max(1)
            train_total += label.size(0)
            train_correct += predicted.eq(label).sum().item()
        
        print('TrainAcc: %.3f%% (%d/%d)' % (
             100. * train_correct / train_total, train_correct, train_total))

        #test
        model2.eval()
        test_correct = 0
        test_total = 0
        with pt.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model2(inputs).to(device)
                loss = lossfunc(outputs, targets).to(device)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'TestAcc: %.3f%% (%d/%d)'
                  % ( 100. * test_correct / test_total, test_correct, test_total))
        test_acc_new = 100. * test_correct / test_total

        if test_acc_new > test_acc:            
            pt.save(model2.state_dict(), "./22_pruning/pruning_" + str(rounds + 1) + ".pt")
            test_acc = test_acc_new
            model2.load_state_dict(pt.load("./22_pruning/pruning_" + str(rounds + 1) + ".pt"))
        else:
            model2.load_state_dict(pt.load("./22_pruning/pruning_" + str(rounds + 1) + ".pt"))

        

    print_sparse(model2)
    
# Train the block pruned model without further pruning
def train_(model1):
    model2 = model1.to(device)
    print("before fine-tuning")
    test_acc=test_model(model2, testloader)
    print_sparse(model2)
    print("begin training")        
    model2.train()
    train_correct = 0
    train_total = 0
    optimizer = pt.optim.Adam(model2.parameters(), lr=0.0001,weight_decay=5e-4)
    lossfunc = pt.nn.CrossEntropyLoss()
    for i, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model2(data).to(device)
        loss = lossfunc(outputs, label).to(device)
        loss.backward()
        grad_puring(model2)
        optimizer.step()
        _, predicted = outputs.max(1)
        train_total += label.size(0)
        train_correct += predicted.eq(label).sum().item()
    
    print('TrainAcc: %.3f%% (%d/%d)' % (
    100. * train_correct / train_total, train_correct, train_total))

    train_acc = 100. * train_correct / train_total


    
    pt.save(model2.state_dict(), "./22_pruning/pruning_0.pt")
    model2.load_state_dict(pt.load("./22_pruning/pruning_0.pt"))

    #test
    model2.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with pt.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model2(inputs).to(device)
            loss = lossfunc(outputs, targets).to(device)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'TestAcc: %.3f%% (%d/%d)'
              % ( 100. * test_correct / test_total, test_correct, test_total))
    test_acc=100. * test_correct / test_total

    train_epoch=1
    while train_epoch < 100:
        train_epoch = train_epoch + 1
        print("training epoch is "+str(train_epoch))        
        model2.train()

        train_correct = 0
        train_total = 0

        optimizer = pt.optim.Adam(model2.parameters(), lr=0.0001, weight_decay=5e-4)
        lossfunc = pt.nn.CrossEntropyLoss()
        for i, (data, label) in enumerate(trainloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model2(data).to(device)
            loss = lossfunc(outputs, label).to(device)
            loss.backward()
            grad_puring(model2)
            optimizer.step()
            _, predicted = outputs.max(1)
            train_total += label.size(0)
            train_correct += predicted.eq(label).sum().item()
        
        print('TrainAcc: %.3f%% (%d/%d)' % (
            100. * train_correct / train_total, train_correct, train_total))
        
        

        #test
        model2.eval()

        test_correct = 0
        test_total = 0

        with pt.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model2(inputs).to(device)
                loss = lossfunc(outputs, targets).to(device)

                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'TestAcc: %.3f%% (%d/%d)'
                  % (100. * test_correct / test_total, test_correct, test_total))
        test_acc_new = 100. * test_correct / test_total


        if test_acc_new > test_acc:
            pt.save(model2.state_dict(), './22_pruning/pruning_0.pt')
            
            test_acc = test_acc_new
        else:
            model2.load_state_dict(pt.load("./22_pruning/pruning_0.pt"))
        
    print_sparse(model2)
    
    return test_acc_new





#This file is actually the unpruned baseline model file,
#just renamed to facilitate iterative pruning
model.load_state_dict(pt.load("./22_pruning/pruning_-1.pt"))


_ = test_model(model,testloader)


print_sparse(model)

rounds=-1
#When the number of rounds is 15, the density is 2.8%.
while rounds<15:
    iterative_pruning_and_training(rounds,2,2)
    rounds=rounds+1
'''

model3=model
m=train_(model3)
'''
