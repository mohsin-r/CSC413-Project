import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer

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
    
# Hyperparameters
input_size = 108
hidden_size = 64
num_layers = 2
batch_size = 64
learning_rate = 0.005
num_epochs = 20
sequence_length = 5

# Create datasets and data loaders
train_dataset = TemperatureDataset(train_data, sequence_length)
val_dataset = TemperatureDataset(val_data, sequence_length)
test_dataset = TemperatureDataset(test_data, sequence_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    total_loss = 0

    # Training loop
    model.train()  
    for batch in train_dataloader:
        inputs, target = batch['sequence'], batch['target']
        output = model(inputs)
        loss = criterion(output, target.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluation on validation data
    val_total_loss = 0
    model.eval()
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_inputs, val_target = val_batch['sequence'], val_batch['target']
            val_output = model(val_inputs)
            val_loss = criterion(val_output, val_target.unsqueeze(1))
            val_total_loss += val_loss.item()

    # Print average training and validation loss for the epoch
    average_loss = total_loss / len(train_dataloader)
    average_val_loss = val_total_loss / len(val_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f} , Validation Loss: {average_val_loss:.4f}")

# Evaluation on test data
test_total_loss = 0
model.eval()
with torch.no_grad():
    for test_batch in test_dataloader:
        test_inputs, test_target = test_batch['sequence'], test_batch['target']
        test_output = model(test_inputs)
        test_loss = criterion(test_output, test_target.unsqueeze(1))
        test_total_loss += test_loss.item()

# Print average test loss
average_test_loss = test_total_loss / len(test_dataloader)
print(f"Test Loss: {average_test_loss:.4f}")