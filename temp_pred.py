import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from tqdm import tqdm
class TemperatureDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.seq_len = sequence_length
        self.data = data.to_numpy()

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return {'sequence': x, 'target': y}
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_kernel_size, hidden_size, num_layers=1):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=cnn_kernel_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = x.permute(0, 2, 1)  
        x = self.cnn(x)
        x = x.permute(0, 2, 1) 
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  
        return out

class RegularizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(RegularizedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, input_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.float()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class RegularizedCNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_kernel_size, hidden_size, num_layers=1, dropout=0):
        super(RegularizedCNNLSTM, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=cnn_kernel_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, input_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.float() 
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
def accuracy(targs, preds, allowance):
    return torch.mean((torch.abs(targs - preds) <= allowance).float()).item()

def train(model, device, train_data, val_data, learning_rate=0.005, batch_size=64, num_epochs=10, seq_len=5, plot=True, file=''):
    train_dataset = TemperatureDataset(train_data, seq_len)
    val_dataset = TemperatureDataset(val_data, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_inputs = None
    val_targets = None

    for val_batch in val_loader:
        if val_inputs is None:
            val_inputs = val_batch['sequence']
        else:
            val_inputs = torch.cat((val_inputs, val_batch['sequence']), 0)
        if val_targets is None:
            val_targets = val_batch['target']
        else:
            val_targets = torch.cat((val_targets, val_batch['target']), 0)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)
    model.to(device)
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        model.train()
        for batch in train_loader:
            inputs, target = batch['sequence'].to(device), batch['target'].to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.float()[:36], target.float()[:36])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if plot:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_inputs)
                val_loss = criterion(val_preds.float()[:36], val_targets.float()[:36])
                v1, v3, v5 = accuracy(val_targets[:36], val_preds[:36], 1), accuracy(val_targets[:36], val_preds[:36], 3), accuracy(val_targets[:36], val_preds[:36], 5)
                train_losses.append(total_loss / len(train_loader))
                val_losses.append(val_loss.item())

    model.eval()
    with torch.no_grad():
        val_preds = model(val_inputs)
        val_loss = criterion(val_preds.float()[:36], val_targets.float()[:36]) 
        v1, v3, v5 = accuracy(val_targets[:36], val_preds[:36], 1), accuracy(val_targets[:36], val_preds[:36], 3), accuracy(val_targets[:36], val_preds[:36], 5)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses[1:], label='Training Loss')
        plt.plot(val_losses[1:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'temp/loss/{file}.jpg')
        with torch.no_grad():
            average_pred = torch.mean(val_preds.cpu(), dim=0)
            average_target = torch.mean(val_targets.cpu(), dim=0)
        plt.figure(figsize=(10, 5))
        plt.scatter(train_data.columns[:36], average_pred[:36], label='Average Predictions')
        plt.scatter(train_data.columns[:36], average_target[:36], label='Average Targets')
        plt.plot(train_data.columns.to_numpy()[:36], average_pred[:36], marker='o', linestyle='-', color='blue')
        plt.plot(train_data.columns.to_numpy()[:36], average_target[:36], marker='o', linestyle='-', color='orange')
        plt.xlabel('City')
        plt.ylabel('Temperature (K)')
        plt.title('Average Predicted City Temperatures vs Actual across Cities')
        plt.xticks(rotation=90)
        plt.legend()
        plt.savefig(f'temp/pred/{file}.jpg')
        return val_loss, v1, v3, v5
    return val_loss

if __name__ == '__main__':
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(device)

    temp_data = pd.read_csv("data/temperature.csv")
    wind_data = pd.read_csv("data/wind_speed.csv")
    pressure = pd.read_csv("data/pressure.csv")

    temp_data['datetime'] = pd.to_datetime(temp_data['datetime'])
    wind_data['datetime'] = pd.to_datetime(wind_data['datetime'])
    pressure['datetime'] = pd.to_datetime(pressure['datetime'])

    # Concatenate and sort data by datetime
    merged_data = pd.concat([temp_data, wind_data.drop(columns=['datetime']), pressure.drop(columns=['datetime'])], axis=1)
    merged_data.sort_values(by='datetime', inplace=True)

    # Splitting the data into training, validation, and test sets based on time periods
    train_data = merged_data[(merged_data['datetime'] >= '2012-01-01') & (merged_data['datetime'] < '2016-01-01')].drop(columns=['datetime'])
    val_data = merged_data[(merged_data['datetime'] >= '2016-01-01') & (merged_data['datetime'] < '2017-01-01')].drop(columns=['datetime'])
    test_data = merged_data[(merged_data['datetime'] >= '2017-01-01') & (merged_data['datetime'] < '2018-01-01')].drop(columns=['datetime'])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    val_data = pd.DataFrame(imputer.fit_transform(val_data), columns=val_data.columns)
    test_data = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)

    # Constants
    input_size = 108
    num_layers = 2
    plot_every = 200

    # Hyperparameters to tune
    best_hidden_size, best_batch_size, best_learning_rate, best_num_epochs, best_seq = [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]
    best_kernel_size, best_dropout = [0,0], [0,0]
    best_v1, best_v2, best_v3, best_v4 = torch.inf, torch.inf, torch.inf, torch.inf

    tune = False
    # Tuning using grid search
    if tune: 
        for sequence_length in [10, 20, 30]:
            for hidden_size, batch_size in [(25,25), (50, 10)]:
                for learning_rate, num_epochs in tqdm([(0.01, 30), (0.1, 15)]):
                    print(f"Testing {hidden_size, batch_size, learning_rate, num_epochs}")
                    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
                    vl1 = train(model, device, train_data, val_data, learning_rate, batch_size, num_epochs, sequence_length, False)
                    if vl1 < best_v1:
                            best_v1, best_hidden_size[0], best_batch_size[0], best_learning_rate[0], best_num_epochs[0], best_seq[0] = vl1, hidden_size, batch_size, learning_rate, num_epochs, sequence_length

                    for kernel_size in [5,7]:
                        model2 = CNNLSTMModel(input_size, kernel_size, hidden_size, num_layers).to(device)
                        vl2 = train(model2, device, train_data, val_data, learning_rate, batch_size, num_epochs, sequence_length, False)
                        if vl2 < best_v2:
                            best_v2 = vl2
                            best_hidden_size[1], best_batch_size[1], best_learning_rate[1], best_num_epochs[1], best_kernel_size[0] = hidden_size, batch_size, learning_rate, num_epochs, kernel_size

                        for dropout in [0.01, 0.1]:
                            model4 = RegularizedCNNLSTM(input_size, kernel_size, hidden_size, num_layers, dropout).to(device)
                            vl4 = train(model4, device, train_data, val_data, learning_rate, batch_size, num_epochs, sequence_length, False)
                            if vl4 < best_v4:
                                best_v4 = vl4
                                best_hidden_size[3], best_batch_size[3], best_learning_rate[3], best_num_epochs[3], best_kernel_size[1], best_dropout[1] = hidden_size, batch_size, learning_rate, num_epochs, kernel_size, dropout

                    for dropout in [0.01, 0.1]:
                        model3 = RegularizedLSTM(input_size, hidden_size, num_layers, dropout).to(device)
                        vl3 = train(model3, device, train_data, val_data, learning_rate, batch_size, num_epochs, sequence_length, False)
                        if vl3 < best_v3:
                            best_v3 = vl3
                            best_hidden_size[2], best_batch_size[2], best_learning_rate[2], best_num_epochs[2], best_dropout[0] = hidden_size, batch_size, learning_rate, num_epochs, dropout                
        print(f"""Best hyperparameters are: 
          \n Losses = {best_v1, best_v2, best_v3, best_v4}
          \n hidden_size = {best_hidden_size}, 
          \n batch_size = {best_batch_size}, 
          \n learning_rate = {best_learning_rate}, 
          \n num_epochs = {best_num_epochs},
          \n kernel_size = {best_kernel_size},
          \n dropout = {best_dropout},

          \n seq_len = {best_seq}""")
    
    else:
        best_hidden_size, best_batch_size, best_learning_rate, best_num_epochs, best_seq = [50, 20, 50, 20], [10, 25, 10, 25], [0.1, 0.1, 0.01, 0.1],  [15,15,30,15], [20, 30, 30, 20]
        best_kernel_size, best_dropout = [5,7], [0.1,0.01]

    model = LSTMModel(input_size, best_hidden_size[0], num_layers).to(device)
    model2 = CNNLSTMModel(input_size, best_kernel_size[0], best_hidden_size[1], num_layers).to(device)
    model3 = RegularizedLSTM(input_size, best_hidden_size[2], num_layers, best_dropout[0]).to(device)
    model4 = RegularizedCNNLSTM(input_size, best_kernel_size[1], best_hidden_size[3], num_layers, best_dropout[1]).to(device)
    r1 = train(model, device, train_data, val_data, best_learning_rate[0], best_batch_size[0], best_num_epochs[0], best_seq[0],  True, 'Vanilla_comp')
    r2 = train(model2, device, train_data, val_data, best_learning_rate[1], best_batch_size[1], best_num_epochs[1], best_seq[1], True, 'CNN_comp')
    r3 = train(model3, device, train_data, val_data, best_learning_rate[2], best_batch_size[2], best_num_epochs[2], best_seq[2], True, 'Dropout_comp')
    r4 = train(model4, device, train_data, val_data, best_learning_rate[3], best_batch_size[3], best_num_epochs[3], best_seq[3], True, 'CNNDropout_comp')

    print(f"""Best Performance evaluation on Validation data:\n\n
          ========================================\n\n
          Model 1: Loss = {r1[0]}, Acc(1,3,5-deg) = {r1[1], r1[2], r1[3]} \n
          Model 2: Loss = {r2[0]}, Acc(1,3,5-deg) = {r2[1], r2[2], r2[3]} \n
          Model 3: Loss = {r3[0]}, Acc(1,3,5-deg) = {r3[1], r3[2], r3[3]} \n
          Model 4: Loss = {r4[0]}, Acc(1,3,5-deg) = {r4[1], r4[2], r4[3]} \n """)