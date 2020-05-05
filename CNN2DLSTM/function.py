import torch

class CNN2DLSTMTraintest():

    def __init__(self, device=None, encoder2Dnet=None, decoderltsm=None, loss=None, optimizer=None):
        super().__init__()
        if encoder2Dnet is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.encoder2Dnet = encoder2Dnet

        if decoderltsm is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.decoderltsm = decoderltsm
        
        self.device = device if device is not None else torch.device('cpu')
        self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss().to(device)

        cnn_params = list(encoder2Dnet.parameters()) + list(decoderltsm.parameters())
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(cnn_params, lr=0.001, momentum=0.4, nesterov=True)
        encoder2Dnet = encoder2Dnet.to(device)
        decoderltsm = decoderltsm.to(device)

    def train(self, trainset):
        self.encoder2Dnet.train()
        self.decoderltsm.train()
        running_loss = 0.0
        total = 0
        for inputs, targets in trainset:
            inputs = self.resizeInputforconv2DLSTM(inputs)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.encoder2Dnet.zero_grad()
            self.decoderltsm.zero_grad()
            outputs = self.encoder2Dnet(inputs)
            outputs = self.decoderltsm(outputs)
            l = self.loss(outputs, targets)
            l.backward()
            self.optimizer.step()
            running_loss += l.item()
            total += targets.size(0)
        return total, running_loss
       

    def test(self, testset):
        self.encoder2Dnet.eval()
        self.decoderltsm.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testset:
                inputs = self.resizeInputforconv2DLSTM(inputs)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.encoder2Dnet.zero_grad()
                self.decoderltsm.zero_grad()
                outputs = self.encoder2Dnet(inputs)
                outputs = self.decoderltsm(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return total, correct

    def resizeInputforconv2DLSTM(self, inputs):
        inputs = torch.cat((inputs[0][0], inputs[0][1]), dim=3).unsqueeze(dim=0)
        return inputs
