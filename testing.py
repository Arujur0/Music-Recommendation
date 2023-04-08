import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from DataSource import Datasource
from models import Emotional

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
testset = Datasource("archive", train = False)
testloader = DataLoader(dataset=testset, batch_size = 32, shuffle=False)
output_labels = []
true_labels = []
load = True
cl = Emotional(1).to(device)
if load:
        print('loading model')
        cl.load_state_dict(torch.load('checkpoints\emotionnet_11.onnx'))
cl.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = cl(inputs)
        true_labels.append(labels)
        _, predicted = torch.max(outputs.data, 1)
        output_labels.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(correct)
    print(total)
    print('[%d epoch] Accuracy of the network on the validation images: %d %%' % 
        (1, 100 * correct / total)
        )