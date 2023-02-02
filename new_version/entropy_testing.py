import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import loss_ce, HLoss

criterion1 = loss_ce()
criterion2 = HLoss()

# 
# print('random_vector: ', random_vector)
# random_vector: tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], requires_grad=True)

# random_vector = random_vector / random_vector.sum()
# print('random_vector/ random_vector.sum(): ', random_vector)
# random_vector/ random_vector.sum(): tensor([0.0634, 0.0091, 0.0538, 0.0870, 0.1370, 0.1771, 0.1082, 0.1669, 0.0994, 0.0982], grad_fn=<DivBackward0>)

# n = 10
# one_hot_vector = torch.tensor([1/n, 1/n, 1/n, 1/n, 1/n, 1/n, 1/n, 1/n, 1/n, 1/n], dtype=torch.float32, requires_grad=True)
# print('one_hot_vector: ', one_hot_vector)

# n = 20
x = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, requires_grad=True)
# x = random_vector = torch.rand(10, requires_grad=True)
x = x.unsqueeze(0)
# Step by Step Entropy Loss
a = F.softmax(x, dim=-1) 
b = F.log_softmax(x, dim=-1)
c = a * b
d = -1.0 * c.sum()

print('x= ', x)
print('a= ', a)
print('b= ', b)
print('c= ', c)
print('d= ', d)



# one_hot_vector:  tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000], requires_grad=True)

# loss = criterion1(one_hot_vector.unsqueeze(0), torch.tensor([2]))
# print('loss: ', loss)
# # loss:  tensor(2.3026, grad_fn=<NllLossBackward0>)

# loss1 = criterion1(one_hot_vector.unsqueeze(0), one_hot_vector.unsqueeze(0))
# print('loss1: ', loss1)
# # loss1:  tensor(2.3026, grad_fn=<DivBackward1>)

# loss2 = criterion2(one_hot_vector.unsqueeze(0))
# print('loss2: ', loss2)
# # loss2:  tensor(2.3026, grad_fn=<MulBackward0>)




# one_hot_vector = torch.nn.functional.one_hot(torch.tensor(2), num_classes=10)
# print('one_hot_vector: ', one_hot_vector)
# loss = criterion1(random_vector.unsqueeze(0), one_hot_vector)