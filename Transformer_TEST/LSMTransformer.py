import torch
import torch.nn as nn

# LSTM model
class LSMLSTM(nn.Module):
    def __init__(self, seq_length, feature_num, hidden_size, output_size, lstm_layer_num=1):
        super(LSMLSTM, self).__init__()
        self.seq_length = seq_length
        self.feature_num = feature_num
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(feature_num, hidden_size, num_layers=lstm_layer_num, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(seq_length*hidden_size, seq_length*int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(seq_length*int(hidden_size/2), seq_length*output_size),
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = output.contiguous().view(output.size()[0], -1)
        # print(output[0])
        output = self.fc(output)  # 使用所有时间步的隐藏状态进行预测
        out = output.view(-1, self.seq_length, self.output_size)
        return out






