import torch.nn as nn

loss = nn.CrossEntropyLoss()

print(loss.__class__.__name__)