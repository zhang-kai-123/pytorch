import torch
import numpy as np
x = torch.arange(1, 3).view(1,2)  #[1,2]
print(x)
y = torch.arange(1, 4).view(3,1)
print(y)
print(x+y)


