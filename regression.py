# coding：utf-8
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 0表示列，1表示行
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x.shape)
print(x)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y =Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.prediction = torch.nn.Linear(n_hidden, n_output)

    def reg_forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.prediction(x)
        return x


net = Net(1, 10, 1)
print(net)

# plt.ion()
# plt.show()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net.reg_forward(x)
    # prediction = net(x)
    loss = loss_func(prediction, y)
    fig = plt.gcf()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        # plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        print('迭代次数：', t)
        plt.pause(0.1)
        save_path = "C:\\Users\\asus\\Desktop\\PytorchNotes\\regress_predic\\predic_{num}.png".format(num=t)
        fig.savefig(save_path)
    # plt.ioff()

    # plt.show()

