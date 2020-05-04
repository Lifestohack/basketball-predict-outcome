import torch

class CNN3DTraintest():

    def __init__(self, device=None, network=None, loss=None, optimizer=None):
        super().__init__()
        if network is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.network = network
        self.device = device if device is not None else torch.device('cpu')
        self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)

    def train(self, trainset):
        self.network.train()
        running_loss = 0.0
        total = 0
        for inputs, targets in trainset:
            inputs = self.resizeInputforconv3D(inputs)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.network(inputs)
            self.network.zero_grad()
            l = self.loss(outputs, targets)
            l.backward()
            self.optimizer.step()
            running_loss += l.item()
            total += targets.size(0)
        return total, running_loss
       

    def test(self, testset):
        self.network.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testset:
                inputs = self.resizeInputforconv3D(inputs)
                inputs = inputs.to(self.device)
                target = targets.to(self.device)
                outputs = self.network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return total, correct

    def resizeInputforconv3D(self, inputs):
        inputs = torch.cat((inputs[0][0], inputs[0][1]), dim=3)
        frame = inputs.shape[0]
        channel = inputs.shape[1]
        height = inputs.shape[2]
        width = inputs.shape[3]
        inputs = inputs.view(-1, channel, frame, height, width)
        return inputs
