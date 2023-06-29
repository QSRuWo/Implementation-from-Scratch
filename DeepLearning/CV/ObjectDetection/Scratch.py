import torch.nn as nn

m = 'Conv'

m_ = eval(m) if isinstance(m, str) else m

print(m_)

