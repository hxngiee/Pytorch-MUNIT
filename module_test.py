import torch


x = torch.tensor([1,2,3])
x = x.repeat(4,2)
print(x.shape)
