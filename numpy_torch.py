import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
torch_to_array = torch_data.numpy()

print(
    '\nnp_data', np_data,
    '\ntorch_data', torch_data,
    '\ntorch_to_array', torch_to_array)