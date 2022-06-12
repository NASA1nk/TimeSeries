import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_output):
        super(LSTM, self).__init__()
        self.model_type = 'LSTM'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.predict = nn.Linear(self.hidden_size, num_output)


    def forward(self, x):
        # x为一维时间序列，因此输入x的维度为(1,batch_size,input_size)，其中input_size就是划分的训练窗口大小
        x = x.unsqueeze(0)
        # h0和c0也可以不指定，默认值即全为0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        output, _ = self.lstm(x,(h0, c0))
        output = self.predict(output[0])
        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = 30
    hidden_size = 128
    num_layers = 3
    num_output = 7
    epochs = 10
    lr = 0.0001
    model = LSTM(input_size,hidden_size,num_layers,num_output)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    train_loss_all = []
    for epoch in range(epochs):
        train_loss = 0
        train_num = 0
        x_seq, y_label
        for step, (x_seq, y_label) in enumerate(train_loader):
            x_seq = x_seq.to(device)
            y_label = y_label.to(device)
            output = model(x_seq)
            loss = loss_func(output, y_label)
            # 素质三连
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_seq.size(0)
            train_num += x_seq.size(0)
        print(f'Epoch {epoch+1} / {max_epoch} | Loss : {train_loss/train_num}')
        train_loss_all.append(train_loss / train_num)
    
    predict = []
    labels = []
    steps = 10
    for step in range(steps):
        test_seq = torch.FloatTensor(predict[-time_window:]).to(device)
        with torch.no_grad():
            model.hidden_state = (torch.zeros(1, 1, model.hidden_size).to(device), 
                              torch.zeros(1, 1, model.hidden_size).to(device))
            predict.append(model(test_seq).item())
    loss = loss_func(torch.tensor(predict[-steps:]), torch.tensor(test_y))
    print(f"Performance on test range: {loss}")