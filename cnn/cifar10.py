import torch
import torchvision
import torchvision.transforms as transforms
from seiya.cnn.LeNet5 import LeNet5
from seiya.cnn.Inception import SimpleInception
import torch.optim as optim
import os
from argparse import ArgumentParser
from deep_rl.util.save_restore import SaveRestoreService

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--data-dir", help="output_dir", required=True)
    args = parser.parse_args()
    data_dir = args.data_dir

    #net = LeNet5()
    net = SimpleInception()
    cross_entropy = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    '''
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(), # chain 2 transforms together using list.
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    record_dir = os.path.join(data_dir, "inception")
    os.makedirs(record_dir, exist_ok=True)
    network_dict = {"inception": net}

    sr_service = SaveRestoreService(record_dir, network_dict)
    sr_service.restore()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for epoch in range(200):  # loop over the dataset multiple times
        cnt = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        if epoch % 10 == 9:
            test(net)
            sr_service.save()

    print('Finished Training')


