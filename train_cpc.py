import torch
import torch.nn as nn
import resnet
from UCF101 import UCF101Dataset
from torch.utils.data import DataLoader


class CPC_net(nn.Module):

    def __init__(self, **kwargs):
        super(CPC_net, self).__init__()
        self.resnet = resnet.resnet10(**kwargs)
        self.fc = nn.Linear(101, 101)

    def forward(self, x, is_train):
        if is_train:
            index = int(x.size(2) / 2)
            x_1 = x[:, :, 0:index, :, :]
            x_2 = x[:, :, index:, :, :]

            out1 = self.resnet(x_1)
            out2 = self.resnet(x_2)

            output = torch.mm(self.fc(out2), out1.t())
        else:
            output = self.resnet(x)

        return output


def NCELoss(input):
    input = torch.exp(input)

    sum = torch.sum(input, 0)

    loss = 0
    for i in range(input.size(1)):
        loss += torch.log(input[i, i] / sum[i])
    return -(loss / input.size(1))


ROOT_PATH = 'F:/Dataset/UCF101/ucfTrainTestlist/'
DATA_PATH = 'F:/Dataset/UCF101/JPEG/'

cpc_net = CPC_net(sample_size=224, sample_duration=16, num_classes=101)

print(cpc_net)

optimizer = torch.optim.Adam(cpc_net.parameters(), lr=0.001)   # optimize all cnn parameters
loss_func = nn.L1Loss()

dataset = UCF101Dataset(ROOT_PATH=ROOT_PATH, DATA_PATH=DATA_PATH,
                            is_training=True, transforms=None, num_segments=16,
                            is_multiple_frames=True, tencrop_mode=False)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

# training and testing
for epoch in range(1):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cpc_net(b_x.reshape(8, 3, -1, 224, 224), True)
        loss = NCELoss(output)
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

        print('Epoch: ', epoch, '| step: ', step, '| train loss: %.4f' % loss.data.numpy())
        break


test_data = UCF101Dataset(ROOT_PATH=ROOT_PATH, DATA_PATH=DATA_PATH,
                            is_training=False, transforms=None, num_segments=16,
                            is_multiple_frames=True, tencrop_mode=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

right_num = 0
for step, (b_x, b_y) in enumerate(test_loader):
    output = cpc_net(b_x.reshape(8, 3, -1, 224, 224), False)
    pred_y = torch.max(output, 1)[1].data.numpy()
    right_num += (pred_y == b_y.data.numpy()).astype(int).sum()
    break

accuracy = float(right_num) / len(test_data)
print('size of TestSet: %d' % len(test_data))
print('test_accuracy: %.2f' % accuracy)