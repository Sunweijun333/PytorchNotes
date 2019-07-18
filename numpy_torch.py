import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
torch_to_array = torch_data.numpy()
# data = [-1, -2, 1, 1]
# tensor = torch.FloatTensor(data)   # 32 bit
data =[[1, 2], [3, 4]]
data = np.array(data)
tensor = torch.FloatTensor(data)
print(
    '\nnp_data:', np_data,
    '\ntorch_data:', torch_data,
    '\ntorch_to_array:', torch_to_array,
    '\ndata:', np.abs(data),
    '\ntensor:', torch.abs(tensor),
    '\nnumpy:', np.matmul(data, data),
    '\ntorch:', torch.mm(tensor, tensor),
    # '\ntorch:', tensor.dot(tensor)   # error
    '\ntorch:', data.dot(data))