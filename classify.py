# coding：utf-8
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
# 按行拼接
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)   # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), 0).type(torch.LongTensor)    # LongTensor = 64-bit integer

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1])
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def reg_forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(2, 10, 2)
print(net)
#
plt.ion()
plt.show()
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
#
for t in range(100):
    out = net.reg_forward(x)
    # prediction = net(x)
    loss = loss_func(out, y)
    print(out.size())
    print(y.size())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        plt.cla()
        # 按照维度选取最大值 0表示列，1表示行，返回最大值及其索引值
        prediction = torch.max(F.softmax(out), 1)[1]
        # squeeze 进行压缩
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        print('迭代次数：', t)
        plt.pause(0.1)
#         save_path = "C:\\Users\\asus\\Desktop\\PytorchNotes\\regress_predic\\predic_{num}.png".format(num=t)
#         fig.savefig(save_path)

plt.ioff()
plt.show()

