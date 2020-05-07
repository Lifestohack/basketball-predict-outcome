import torch

class OPTICALCONV3DTraintest():

    def __init__(self, device=None, cnn3d=None, opticalcnn3d=None, twostream=None,loss=None, optimizer=None):
        super().__init__()
        if cnn3d is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.cnn3d = cnn3d
        if opticalcnn3d is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.opticalcnn3d = opticalcnn3d
        if twostream is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.twostream = twostream
        self.device = device if device is not None else torch.device('cpu')
        self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss().to(device)
        cnn_params = list(self.cnn3d.parameters()) + list(self.opticalcnn3d.parameters()) + list(self.twostream.parameters())
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(cnn_params, lr=0.001, momentum=0.4, nesterov=True)
        self.cnn3d = self.cnn3d.to(device)
        self.opticalcnn3d = self.opticalcnn3d.to(device)
        self.twostream = self.twostream.to(device)

    def train(self, trainset):
        self.cnn3d.train()
        self.opticalcnn3d.train()
        self.twostream.train()
        running_loss = 0.0
        total = 0
        outputslist = []
        #conv3d
        for inputs, targets in trainset:
            inputs = self.resizeInputforconv3D(inputs)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.cnn3d.zero_grad()
            outputs = self.cnn3d(inputs)
            outputslist.append(outputs)

        #opticalconv3d
        for inputs, targets in trainset:
            inputs = self.resizeInputforconv3D(inputs)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.opticalcnn3d.zero_grad()
            outputs = self.opticalcnn3d(inputs)
            outputslist.append(outputs)

        #outputs = torch.stack(outputslist, dim=0)
        print(outputslist[0].shape)
        print(outputslist[1].shape)
        self.twostream.zero_grad()
        outputs = self.twostream(outputs)
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
        #inputs = torch.cat((inputs[0][0], inputs[0][1]), dim=3)
        #inputs = inputs.transpose(2, 3) # transpose decreased the accuracy from 96.4 to 89
        frame = inputs.shape[1]
        channel = inputs.shape[2]
        height = inputs.shape[3]
        width = inputs.shape[4]
        inputs = inputs.view(-1, channel, frame, height, width)
        return inputs
