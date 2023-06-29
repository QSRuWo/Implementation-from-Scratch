import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch.utils import data

'''
In this file, we try to use a MLP to predict a sequential information.
'''

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
plt.plot(time, x)
plt.xlabel('time')
plt.ylabel('x')
plt.show()

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i : T - tau + i]
labels = x[tau:].reshape(-1,1)
# print(labels.shape)

batch_size, n_train = 16, 600

def load_array(data_arrays, batch_size, is_Train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_Train)

train_loader = load_array((features[:n_train], labels[:n_train]), batch_size=batch_size, is_Train=True)

def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)

net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
net.apply(init_weights)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

epochs = 5

for epoch in range(epochs):
    net.train()
    running_loss = 0.
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch} loss {running_loss / batch_size}')

# Check the prediction
pred = net(features)
plt.figure(figsize=(10,5))
plt.plot(time, x, label='True label', color='blue')
plt.plot(time[tau:], pred.detach().numpy(), label='Prediction', color='purple')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()
plt.show()

# multi-step prediction
multistep_preds = torch.zeros((T,))
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau : i].reshape(1, -1))

plt.figure(figsize=(10,5))
plt.plot(time, x, label='True label', color='blue')
plt.plot(time[tau:], pred.detach().numpy(), label='Prediction', color='purple')
plt.plot(time[n_train + tau:], multistep_preds[n_train + tau:].detach().numpy(), label='Multistep-Prediction', color='green')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()
plt.show()