import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer


temp_data = pd.read_csv("data/temperature.csv").drop(columns=['datetime'])
wind_data = pd.read_csv("data/wind_speed.csv").drop(columns=['datetime'])
pressure = pd.read_csv("data/pressure.csv").drop(columns=['datetime'])

imputer = SimpleImputer(strategy='mean')  
temp_data_imputed = imputer.fit_transform(temp_data)
wind_data_imputed = imputer.fit_transform(wind_data)
pressure_imputed = imputer.fit_transform(pressure)

# Convert imputed data back to DataFrame
temp_data = pd.DataFrame(temp_data_imputed, columns=temp_data.columns)
wind_data = pd.DataFrame(wind_data_imputed, columns=wind_data.columns)
pressure = pd.DataFrame(pressure_imputed, columns=pressure.columns)


class TemperatureDataset(Dataset):
    def __init__(self, temp_data, wind_data, pressure, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = temp_data.to_numpy()
        self.wind_data = wind_data.to_numpy()
        self.pressure = pressure.to_numpy()

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        temp_sequence = torch.tensor(self.data[idx:end_idx], dtype=torch.float32)
        wind_sequence = torch.tensor(self.wind_data[idx:end_idx], dtype=torch.float32)
        pressure_sequence = torch.tensor(self.pressure[idx:end_idx], dtype=torch.float32)
        combined_sequence = torch.cat((temp_sequence, wind_sequence, pressure_sequence), dim=1)
        
        target = torch.tensor(self.data[end_idx], dtype=torch.float32)[-1]
        return {
            'combined_sequence': combined_sequence,
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
learning_rate = 0.001
num_epochs = 10

# Prepare data
sequence_length = 10 
temp_dataset = TemperatureDataset(temp_data, wind_data, pressure, sequence_length)
dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    for batch in dataloader:
        combined_sequence = batch['combined_sequence']
        target = batch['target'].unsqueeze(1)

        # Forward pass
        output = model(combined_sequence)
        # Calculate loss
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    average_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")