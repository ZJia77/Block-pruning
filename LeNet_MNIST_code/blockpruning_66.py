import torch as pt
import torchvision as ptv
import numpy as np
import math
import lenet
#train the 6*6 block pruning granularity model
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")

#MNIST dataset
train_dataset = ptv.datasets.MNIST("./",download = True,transform = ptv.transforms.Compose([ptv.transforms.ToTensor(),ptv.transforms.Normalize([0.5], [0.5])]))
test_dataset = ptv.datasets.MNIST("./",train = False,transform = ptv.transforms.Compose([ptv.transforms.ToTensor(),ptv.transforms.Normalize([0.5], [0.5])]))
#loader
trainloader = pt.utils.data.DataLoader(train_dataset,shuffle = True,batch_size = 50)
testloader = pt.utils.data.DataLoader(test_dataset,shuffle = False,batch_size = 50)
#LeNet-300-100 model



'''
input:
number1, number2: the block pruning granularity(6)
model1_fc1_tensor, model1_fc2_tensor, model1_fc3_tensor: Weight matrix in each layer

output:
tensor1,tensor2,tensor3: Block-based division structure.
The elements in the output matrix are the average of absolute values of weights in sub-blocks.
This facilitates the selection of sub-blocks to be pruned.
'''
# This function is to compute the sum of weights in each block
# after block division of the weight matrix
def division(number1,number2,model1_fc1_tensor,model1_fc2_tensor,model1_fc3_tensor):
    tensor1 = pt.zeros(math.ceil(300/number1),math.ceil(784/number2))
    tensor2 = pt.zeros(math.ceil(100/number1),math.ceil(300/number2))
    tensor3 = pt.zeros(math.ceil(10/number1),math.ceil(100/number2))
    # the number of weights of each sub-block in every layer
    sumnum1=0
    sumnum2=0
    sumnum3=0
    #compute the average of absolute values of weights in sub-blocks
    #layer1
    for m in range(math.ceil(300/number1)):
        for n in range(math.ceil(784/number2)):            
                for i in range(number1):
                    for j in range(number2):
                        if number1*m+i<=300-1 and number2*n+j<=784-1:                
                            tensor1[m][n]=tensor1[m][n]+pt.abs(model1_fc1_tensor[number1*m+i][number2*n+j])
                            sumnum1=sumnum1+1
                tensor1[m][n]=tensor1[m][n]/sumnum1
                sumnum1=0    
    max_weight1=pt.max(abs(tensor1))
    for m in range(math.ceil(300/number1)):
        for n in range(math.ceil(784/number2)):
            if tensor1[m][n]!=0:
                tensor1[m][n]=tensor1[m][n]/max_weight1

    #layer2            
    for m in range(math.ceil(100/number1)):
        for n in range(math.ceil(300/number2)):
            for i in range(number1):
                    for j in range(number2):            
                        if number1*m+i<=100-1 and number2*n+j<=300-1:
                            tensor2[m][n]=tensor2[m][n]+pt.abs(model1_fc2_tensor[number1*m+i][number2*n+j])  
                            sumnum2=sumnum2+1
            tensor2[m][n]=tensor2[m][n]/sumnum2
            sumnum2=0
    max_weight2=pt.max(abs(tensor2))
    for m in range(math.ceil(100/number1)):
        for n in range(math.ceil(300/number2)):
            if tensor2[m][n]!=0:
                tensor2[m][n]=tensor2[m][n]/max_weight2

    #layer3            
    for m in range(math.ceil(10/number1)):        
        for n in range(math.ceil(100/number2)):
            for i in range(number1):
                    for j in range(number2):
                        if number1*m+i<=10-1 and number2*n+j<=100-1:
                            tensor3[m][n]=tensor3[m][n]+pt.abs(model1_fc3_tensor[number1*m+i][number2*n+j])  
                            sumnum3=sumnum3+1
            tensor3[m][n]=tensor3[m][n]/sumnum3
            sumnum3=0
    max_weight3=pt.max(abs(tensor3))
    for m in range(math.ceil(10/number1)):
        for n in range(math.ceil(100/number2)):
            if tensor3[m][n]!=0:
                tensor3[m][n]=tensor3[m][n]/max_weight3        
    return tensor1,tensor2,tensor3

'''
input: the output of division function
output: the pruned weight matrix
'''
# This function is used to iteratively prune the sub-block
def pruning(new_tensor1,new_tensor2,new_tensor3):    
    #sort the weight matrix in each layer
    sorted_tensor1= np.sort(new_tensor1[new_tensor1!=0],axis=None)
    # 0.2 means the pruning rate in this layer
    cutoff_index1 = np.round(0.2 * sorted_tensor1.size).astype(int)
    cutoff1 = sorted_tensor1[cutoff_index1]
    new_tensor1=np.where(new_tensor1 < cutoff1, 0, new_tensor1)
    
    
    sorted_tensor2= np.sort(new_tensor2[new_tensor2!=0],axis=None)
    # 0.2 means the pruning rate in this layer
    cutoff_index2 = np.round(0.2* sorted_tensor2.size).astype(int)
    cutoff2 = sorted_tensor2[cutoff_index2]
    new_tensor2=np.where(new_tensor2 < cutoff2, 0, new_tensor2)
    
    
    sorted_tensor3= np.sort(new_tensor3[new_tensor3!=0],axis=None)
    # 0.1 means the pruning rate in this layer
    cutoff_index3 = np.round(0.1 * sorted_tensor3.size).astype(int)
    cutoff3 = sorted_tensor3[cutoff_index3]
    new_tensor3=np.where(new_tensor3 < cutoff3, 0, new_tensor3)
    
    return new_tensor1,new_tensor2,new_tensor3

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

# a mask can be set based on the corresponding weights
# and make element-wise product with the weight gradient to ensure
# that the removed weight values are not updated.

def grad_puring(model):
    for i in model.parameters():
        mask = i.clone()
        mask[mask != 0] = 1
        i.grad.data.mul_(mask)

# Output weight matrix in the layer 1
def draw_connection_layer1_num(model):
    t1=model.state_dict()["fc1.weight"].numpy()    
    # The naming convention is "layer_pruning granularity_weight_density.txt"
    filename='./layer1_66_weight_0.028.txt'
    with open(filename,'w') as f:        
        for i in range(300):
            for j in range(784):
                f.write(str(t1[i][j])+" ")
            f.write("\n")
        f.close()
        
# Output weight matrix in the layer 2        
def draw_connection_layer2_num(model):    
    t2=model.state_dict()["fc2.weight"].numpy()    
    filename='./layer2_66_weight_0.028.txt'    
    with open(filename,'w') as f:
        for i in range(100):
            for j in range(300):
                f.write(str(t2[i][j])+" ")
            f.write("\n")
        f.close()
        
# Output weight matrix in the layer 3
def draw_connection_layer3_num(model):
    t3=model.state_dict()["fc3.weight"].numpy()
    filename='./layer3_66_weight_0.028.txt'
    with open(filename,'w') as f:
        for i in range(10):
            for j in range(100):
                f.write(str(t3[i][j])+" ")
            f.write("\n")
        f.close()



#Sparsity evaluation function
def print_sparse(model):
    # the number of weights in the matrix
    total_num = 0
    # the zero number pf weights in the matrix
    total_sparse = 0
    print("-----------------------------------")
    print("Layer sparsity")
    for name,f in model.named_parameters():
        num = f.view(-1).shape[0]
        total_num += num
        sparse = pt.nonzero(f).shape[0]
        total_sparse+= sparse
        print("\t",name,(sparse)/num)
    total = total_sparse/total_num
    print("Total:",total)
    params=list(model.named_parameters())
    num1=params[0][1].view(-1).shape[0]
    sum1=pt.nonzero(params[0][1]).shape[0]
    num2=params[2][1].view(-1).shape[0]
    sum2=pt.nonzero(params[2][1]).shape[0]
    num3=params[4][1].view(-1).shape[0]
    sum3=pt.nonzero(params[4][1]).shape[0]
    num=num1+num2+num3
    thesum=sum1+sum2+sum3
    sp=thesum/num
    print(sp)
    return total



# Train the block pruned model without further pruning
def train_(model1):
    print("before fine-tuning")
    test_acc=test_model(model1,testloader)
    train_acc=train_model(model1,trainloader)        
    print_sparse(model1)
    print("begin training")
    optimizer = pt.optim.Adam(model1.parameters(), 1e-6)
    lossfunc = pt.nn.CrossEntropyLoss()
    for i, (data, label) in enumerate(trainloader):
        data=data.to(device)
        label=label.to(device)        
        outputs = model1(data).to(device)
        loss = lossfunc(outputs, label).to(device)
        loss.backward()
        grad_puring(model1)
        optimizer.step()
        
    test_acc=test_model(model1,testloader)
    train_acc=train_model(model1,trainloader)    
    pt.save(model1.state_dict(),"./66_pruning/pruning_10.pt")
    model1.load_state_dict(pt.load("./66_pruning/pruning_10.pt"))
    train_epoch=1
    
    while train_epoch<1000:
        train_epoch=train_epoch+1
        print("training epoch is "+str(train_epoch))
        optimizer = pt.optim.Adam(model1.parameters(), 1e-6)
        lossfunc = pt.nn.CrossEntropyLoss()
        for i, (data, label) in enumerate(trainloader):
            data=data.to(device)
            label=label.to(device)            
            outputs = model1(data).to(device)
            loss = lossfunc(outputs, label).to(device)
            loss.backward()
            grad_puring(model1)
            optimizer.step()
            
        print("after fine-tuning:")
        test_acc_new=test_model(model1,testloader)
        train_acc_new=train_model(model1,trainloader)
        
        
        if test_acc_new>test_acc:
            pt.save(model1.state_dict(),"./66_pruning/pruning_10.pt")
            test_acc=test_acc_new
            model1.load_state_dict(pt.load("./66_pruning/pruning_10.pt"))
        else:
            model1.load_state_dict(pt.load("./66_pruning/pruning_10.pt"))
        
    print_sparse(model1)
    
    return test_acc_new





'''
input:
rounds: number of rounds of iterative block pruning
        For example, according to the pruning rate in this paper,
        the density reaches 8% at the 10th round of pruning.
        
        The number of rounds facilitates the naming of model saving files.
        
        Special definition: The round of the unpruned baseline model is defined as -1,
        which is convenient for quickly calling the corresponding saving model file
        during iterative pruning.
number1, number2: the block pruning granularity(6)
'''
# iterative pruning function
def iterative_pruning_and_training(rounds , number1 , number2):
    # When the sparsity is not high, the model converges faster.
    # In order to reduce the training time, the number of training is relatively
    # less in the first few rounds of block pruning
    epoch=1000
    model = lenet.net().to(device)    
    model.load_state_dict(pt.load("./66_pruning/pruning_"+str(rounds)+".pt"))
    test_model(model,testloader)
    train_model(model,trainloader)
    print_sparse(model)

    # Get the weight matrix
    model2=model
    model_fc1_tensor=model2.state_dict()["fc1.weight"].float()
    print(model_fc1_tensor.size(),type(model_fc1_tensor))
    model_fc2_tensor=model2.state_dict()["fc2.weight"].float()
    print(model_fc2_tensor.size(),type(model_fc2_tensor))
    model_fc3_tensor=model2.state_dict()["fc3.weight"].float()
    print(model_fc3_tensor.size(),type(model_fc3_tensor))
    #division
    new_tensor1,new_tensor2,new_tensor3=division(number1,number2,model_fc1_tensor,model_fc2_tensor,model_fc3_tensor)        
    #iterative pruning
    new_tensor1,new_tensor2,new_tensor3 = pruning(new_tensor1,new_tensor2,new_tensor3)

    params = list(model2.named_parameters())
    #According to the pruned sub-block, set the weights corresponding to the sub-block position to zero
    for m in range(math.ceil(300/number1)):
        for n in range(math.ceil(784/number2)):
            if new_tensor1[m][n]==0:
                for i in range(number1):
                    for j in range(number2):
                        if number1*m+i<=300-1 and number2*n+j<=784-1:
                            params[0][1].data[number1*m+i][number2*n+j]=0
                    

    for m in range(math.ceil(100/number1)):
        for n in range(math.ceil(300/number2)):
            if new_tensor2[m][n]==0:
                for i in range(number1):
                    for j in range(number2):
                        if number1*m+i<=100-1 and number2*n+j<=300-1:
                            params[2][1].data[number1*m+i][number2*n+j]=0

    for m in range(math.ceil(10/number1)):
        for n in range(math.ceil(100/number2)):
            if new_tensor3[m][n]==0:
                for i in range(number1):
                    for j in range(number2):
                        if number1*m+i<=10-1 and number2*n+j<=100-1:
                            params[4][1].data[number1*m+i][number2*n+j]=0

                
    print("before fine-tuning")
    test_acc=test_model(model2,testloader)
    train_acc=train_model(model2,trainloader)        
    print_sparse(model2)
    print("begin training")
    optimizer = pt.optim.Adam(model2.parameters(), 1e-6)
    lossfunc = pt.nn.CrossEntropyLoss()
    for i, (data, label) in enumerate(trainloader):
        data=data.to(device)
        label=label.to(device)        
        outputs = model2(data).to(device)
        loss = lossfunc(outputs, label).to(device)
        loss.backward()
        grad_puring(model2)
        optimizer.step()
        
        
    test_acc=test_model(model2,testloader)
    train_acc=train_model(model2,trainloader)    
    pt.save(model2.state_dict(),"./66_pruning/pruning_"+str(rounds+1)+".pt")
    model2.load_state_dict(pt.load("./66_pruning/pruning_"+str(rounds+1)+".pt"))
    train_epoch=1
    
    while train_epoch<epoch:
        train_epoch=train_epoch+1
        print("training epoch is "+str(train_epoch))
        optimizer = pt.optim.Adam(model2.parameters(), 1e-6)
        lossfunc = pt.nn.CrossEntropyLoss()
        for i, (data, label) in enumerate(trainloader):
            data=data.to(device)
            label=label.to(device)            
            outputs = model2(data).to(device)
            loss = lossfunc(outputs, label).to(device)
            loss.backward()
            grad_puring(model2)
            optimizer.step()
            
        print("after fine-tuning:")
        test_acc_new=test_model(model2,testloader)
        train_acc_new=train_model(model2,trainloader)
        
        
        if test_acc_new>test_acc:
            pt.save(model2.state_dict(),"./66_pruning/pruning_"+str(rounds+1)+".pt")
            test_acc=test_acc_new
            model2.load_state_dict(pt.load("./66_pruning/pruning_"+str(rounds+1)+".pt"))
        else:
            model2.load_state_dict(pt.load("./66_pruning/pruning_"+str(rounds+1)+".pt"))
        
    print_sparse(model2)
    


model = lenet.net().to(device)
#This file is actually the unpruned baseline model file,
#just renamed to facilitate iterative pruning
model.load_state_dict(pt.load("./66_pruning/pruning_10.pt"))


test_model(model,testloader)
train_model(model,trainloader)
print_sparse(model)

rounds=-1
#When the number of rounds is 15, the density is 2.8%.
while rounds<15:
    iterative_pruning_and_training(rounds,6,6)
    rounds=rounds+1

'''
model3=model
m=train_(model3)
'''

