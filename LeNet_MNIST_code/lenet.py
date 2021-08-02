import torch as pt
import torchvision as ptv

#the model structure
class net(pt.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = pt.nn.Linear(in_features=784, out_features=300)
        self.fc2 = pt.nn.Linear(in_features=300, out_features=100)
        self.fc3 = pt.nn.Linear(in_features=100, out_features=10)
        
    def forward(self, x):
        x = x.view(-1,28*28)        
        dout = pt.nn.functional.relu(self.fc1(x))                
        dout = pt.nn.functional.relu(self.fc2(dout))        
        return pt.nn.functional.softmax(self.fc3(dout))


