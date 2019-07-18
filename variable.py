import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)    # 1/4*x^2 -----> 1/4*2x

# print(tensor)
# print(variable)
# print(t_out)
# print(v_out)

v_out.backward()
print(variable.grad)
print(variable)
print(variable.data)
print(variable.data.numpy)