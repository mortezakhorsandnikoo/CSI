import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm



from resnet  import ResNet, ResidualBlock, Bottleneck



batch_size = 30
num_epochs = 7
learning_rate = 0.002


# load data
data = sio.loadmat('data/aug_train_data.mat')
train_data = data['aug_train_data']
train_label = data['aug_train_label']

# label matrix organized as nSamplex5, where the 1st coloum is the index of personID, the latter 4 are 4 biometrcs
train_label[:, 0] = train_label[:, 0] - 1         # 1--30 -> 0--29

num_train_instances = len(train_data)
# prepare data, nSample x nChannel x width x height
# reshape train data size to nSample x nSubcarrier x 1 x 1
train_data = torch.from_numpy(train_data).type(torch.FloatTensor).view(num_train_instances, 30, 1, 1)
train_label = torch.from_numpy(train_label).type(torch.FloatTensor)
train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# load test data
data = sio.loadmat('data/test.mat')
test_data = data['test_data']
test_label = data['test_label']
test_label[:, 0] = test_label[:, 0] - 1

num_test_instances = len(test_data)
# prepare data, nSample x nChannel x width x height
# reshape test data size to nSample x nSubcarrier x 1 x 1
test_data = torch.from_numpy(test_data).type(torch.FloatTensor).view(num_test_instances, 30, 1, 1)
test_label = torch.from_numpy(test_label).type(torch.FloatTensor)
test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


resnet = ResNet(ResidualBlock, [2, 2, 2, 2], 30)
resnet = resnet.cuda()

criterion1 = nn.CrossEntropyLoss().cuda()
criterion2 = nn.L1Loss().cuda()
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.3)

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    resnet.train()

    scheduler.step()
    # trained_num = 0
    for (samples, labels) in tqdm(train_data_loader):
        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label = resnet(samplesV)

        lossC = criterion1(predict_label[0], labelsV[:, 0].type(torch.LongTensor).cuda())

        lossR1 = criterion2(predict_label[1][:, 0], labelsV[:, 1])
        lossR2 = criterion2(predict_label[1][:, 1], labelsV[:, 2])
        lossR3 = criterion2(predict_label[1][:, 2], labelsV[:, 3])
        lossR4 = criterion2(predict_label[1][:, 3], labelsV[:, 4])

        loss = (lossC + 0.0386*lossR1 + 0.0405*lossR2 + 0.0629*lossR3 + 0.0877*lossR4)/4
        loss.backward()
        optimizer.step()
# #
    resnet.eval()
    correct_t = 0
    for (samples, labels) in tqdm(train_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())
            # labelsV = labelsV.view(-1)

            predict_label = resnet(samplesV)
            prediction = predict_label[0].data.max(1)[1]
            correct_t += prediction.eq(labelsV[:, 0].data.long()).sum()

    print("Training accuracy:", (100*float(correct_t)/num_train_instances))

    trainacc = str(100*float(correct_t)/num_train_instances)[0:6]

    correct_t = 0
    for (samples, labels) in tqdm(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())
            # labelsV = labelsV.view(-1)

        predict_label = resnet(samplesV)
        prediction = predict_label[0].data.max(1)[1]
        correct_t += prediction.eq(labelsV[:, 0].data.long()).sum()

    print("Test accuracy:", (100 * float(correct_t) / num_test_instances))

    testacc = str(100 * float(correct_t) / num_test_instances)[0:6]

    torch.save(resnet, 'weights/resnet18_Train' + trainacc + 'Test' + testacc + '.pkl')








