import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer

class TemperatureDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = data.to_numpy()

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        sequence = torch.tensor(self.data[idx:end_idx], dtype=torch.float32)
        target = torch.tensor(self.data[end_idx][-1], dtype=torch.float32)
        return {
            'sequence': sequence,
            'target': target
        }
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def accuracy(targs, preds, allowance):
    assert targs.shape == preds.shape
    return torch.mean((torch.abs(targs - preds) <= allowance).float()).item()

def train(model, device, train_data, val_data, learning_rate=0.005, batch_size=64, num_epochs=10, seq_len=5, plot_every=50, plot=True, allowance=1.):
    # Create datasets and data loaders
    train_dataset = TemperatureDataset(train_data, seq_len)
    val_dataset = TemperatureDataset(val_data, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.MSELoss()
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
        

    iter_count = 0
    for epoch in range(num_epochs):
        total_loss = 0

        # Training loop
        model.train()  
        for batch in train_loader:
            iter_count += 1
            inputs, target = batch['sequence'].to(device), batch['target'].to(device)
            output = model(inputs)
            loss = criterion(output.to(device), target.unsqueeze(1).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if iter_count % plot_every == 0:
                ta = accuracy(target.to(device), output.squeeze(1).to(device), allowance)
                val_preds = model(val_inputs)
                va = accuracy(val_targets, val_preds.squeeze(1), allowance)
                print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)

        # Evaluation on validation data at end of epoch
        model.eval()
        with torch.no_grad():
            val_preds = model(val_inputs)
            va = accuracy(val_targets, val_preds.squeeze(1), allowance)
            val_loss = criterion(val_preds, val_targets.unsqueeze(1).to(device))

        # Print average training and validation loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_loss:.4f} , Average Validation Loss: {val_loss:.4f}, Validation Accuracy: {va:.4f}")

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

    # Hyperparameters
    input_size = 108
    hidden_size = 64
    num_layers = 2
    batch_size = 64
    learning_rate = 0.005
    num_epochs = 20
    seq_len = 5
    plot_every = 100
    allowance = 5.

    print(f"Accuracy will be tested by checking if the predicted and target temperatures are within {allowance} degrees of each other.")

    # Initialize model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers)
    model = model.to(device)

    train(model, device, train_data, val_data, learning_rate, batch_size, num_epochs, seq_len, plot_every, True, allowance)

    test_dataset = TemperatureDataset(test_data, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # Evaluation on test data
    criterion = nn.MSELoss()
    model.eval()

    test_inputs = None
    test_targets = None

    for test_batch in test_loader:
        if test_inputs is None:
            test_inputs = test_batch['sequence']
        else:
            test_inputs = torch.cat((test_inputs, test_batch['sequence']), 0)
        if test_targets is None:
            test_targets = test_batch['target']
        else:
            test_targets = torch.cat((test_targets, test_batch['target']), 0)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    with torch.no_grad():
        test_preds = model(test_inputs)
        ta = accuracy(test_targets, test_preds.squeeze(1), allowance)
        test_loss = criterion(test_preds, test_targets.unsqueeze(1).to(device))

    # Print average test loss
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {ta:.4f}")
