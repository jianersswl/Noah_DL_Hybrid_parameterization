import torch
import torch.nn as nn

class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DynamicFCNetworkTest(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DynamicFCNetworkTest, self).__init__()
        self.layers = nn.ModuleList()
        # 添加输入层到第一个隐藏层的连接
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # 添加隐藏层之间的连接
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        # 添加最后一个隐藏层到输出层的连接
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        out = self.layers[-1](x)
        return out



class DynamicFCNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DynamicFCNetwork, self).__init__()
        self.layers = nn.ModuleList()
        # 添加输入层到第一个隐藏层的连接
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # 添加隐藏层之间的连接
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        # 添加最后一个隐藏层到输出层的连接
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        out = self.layers[-1](x)
        return out