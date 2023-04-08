import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from scipy.special import softmax
import torch.utils.data as data
from DataSource import Datasource
from models import Emotional
import torch.utils.data as data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
def main(epochs, batch_size, learning_rate, save_freq, data_dir):
    # train dataset and train loader
    datasource = Datasource(data_dir, train=True)
    train_loader = data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)
    # load model
    load = True
    model = Emotional(1).to(device)
    # posenet = model().to(device)
    if load:
        print('loading model')
        model.load_state_dict(torch.load('checkpoints\emotionnet_11.onnx'))
    #loss function
    criterion = nn.CrossEntropyLoss()

    # train the network
    optimizer = optim.Adam(nn.ParameterList(model.parameters()),
                     lr=learning_rate,weight_decay=0.01)

    batches_per_epoch = len(train_loader.batch_sampler)

    for epoch in range(epochs):
        model.train()
        for step, batches in enumerate(train_loader):
            #print(batches[1].shape)
            images, labels = batches
            images = images.to(device)
            labels = labels.to(device)
            predict = model(images)
            loss = criterion(predict, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch : {}, Batch: {} of {}: loss = {}".format(epoch, step+1, batches_per_epoch, loss))
            if (epoch + 1) % 10 == 0:

                save_filename = "epoch007.onnx"
                save_path = os.path.join('checkpoints', save_filename)
                torch.save(model.state_dict(), save_path)
if __name__ == '__main__':

    main(epochs=30, batch_size=48, learning_rate=5e-3, save_freq=20, data_dir="archive")
