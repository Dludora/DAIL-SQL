import torch

a = torch.arange(24).view(1,2,3,4)
b = a.view(1,3,2,4)
c = a.transpose(1, 2)

print(a)
print(b)
print(c)