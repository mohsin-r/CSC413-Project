# A RNN to forecast humidity

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import utils

# just some small data and model as placeholder
# TODO:
#   - create a more complex model
#   - proper training, accuracy functions
#   - train, val, test splits
#   - incorporate other features in model
#   - handle multiple cities of data
#   - handle nans more cleanly

# get only one city's humidity data, cutting out NaNs
all_data = pd.read_csv("data/humidity.csv")
data = torch.tensor(all_data["Vancouver"].dropna().values)

# split the data into one-day periods, starting every 3 hours
# since NaNs were just cut, some sequences may not be just one day
sequenced_data, targets = utils.generate_sequences(data, 24, 3)
small_data = sequenced_data[:20] 
small_targets = targets[:20]

# a basic model 
class HumidityRNN(nn.Module):
    def __init__(self, hidden_size):
        super(HumidityRNN, self).__init__()
        self.rnn = torch.nn.RNN(1, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, X):
        h, out = self.rnn(X)
        return self.fc(torch.squeeze(out, dim=0))

# small training code just to verify it all likely works
model = HumidityRNN(20)
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.MSELoss()
epochs = 100
for i in range(epochs):
    z = model(small_data)
    loss = criterion(z, small_targets)
    loss.backward() # propagate the gradients
    optimizer.step() # update the parameters
    optimizer.zero_grad() # clean up accumulated gradients
    if (i % (epochs/10) == 0):
        print(float(loss))