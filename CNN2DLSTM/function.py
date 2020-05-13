import torch
import time
import sys

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
        #self.encoder2Dnet = self.encoder2Dnet.to(device)
        #self.decoderltsm = self.decoderltsm.to(device)

    def train(self, trainset):
        self.encoder2Dnet.train()
        self.decoderltsm.train()
        running_loss = 0.0
        running_total = 0
        total_time_required = 0
        start = time.time()
        for inputs, targets in trainset:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.encoder2Dnet.zero_grad()
            self.decoderltsm.zero_grad()
            outputs = self.__run(inputs, targets)
            #loss, backpropagation
            l = self.loss(outputs, targets)
            l.backward()
            self.optimizer.step()
            running_loss += l.item()
            running_total += targets.size(0)
            end = time.time()
            time_required = (end-start)
            total_time_required += time_required
            self.__print(time_required, total_time_required, running_total, len(trainset.dataset))
            start = time.time()
            torch.cuda.empty_cache()
        return running_loss

    def test(self, testset):
        self.encoder2Dnet.eval()
        self.decoderltsm.eval()
        running_loss = 0.0
        correct = 0
        running_total_tested = 0
        with torch.no_grad():
            start = time.time()
            total_time_required = 0
            for inputs, targets in testset:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.__run(inputs, targets)
                l = self.loss(outputs, targets)
                running_loss += l.item()
                _, predicted = torch.max(outputs.data, 1)
                running_total_tested += targets.size(0)
                correct += (predicted == targets).sum().item()
                end = time.time()
                time_required = (end-start)
                total_time_required += time_required
                self.__print(time_required, total_time_required, running_total_tested, len(testset.dataset))
                start = time.time()
                torch.cuda.empty_cache()
        return correct, running_loss

    def __run(self, inputs, targets):
        inputs = self.__resize(inputs)
        if len(inputs.shape) == 6:  # if two views then concatenated
            inputs = torch.cat([inputs[0][0], inputs[0][1]], dim=2).unsqueeze(dim=0) #concatenate two view
        outputs = self.encoder2Dnet(inputs) #Encoder
        outputs = self.decoderltsm(outputs) #Decoder
        return outputs

    def __resize(self, inputs):
        return inputs

    def __print(self, time_required, total_time_required, total, num_samples):
        outstr = 'Time required for last sample: {:.2f}sec. Total time: {:.2f}sec.  Total tests: {}/{}'.format(time_required, total_time_required, total, num_samples)
        sys.stdout.write('\r'+ outstr)

