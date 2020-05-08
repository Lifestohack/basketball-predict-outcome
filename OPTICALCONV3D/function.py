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

    def train(self, trainset, trainsetoptical):
        self.cnn3d.train()
        self.opticalcnn3d.train()
        self.twostream.train()
        running_loss = 0.0
        total = 0
        
        for conv3d, optical in zip(trainset, trainsetoptical):
            conv3d_inputs = conv3d[0].to(self.device)
            optical_inputs= optical[0].to(self.device)
            targets = conv3d[1].to(self.device)
            
            conv3d_inputs = self.resizeInputforconv3D(conv3d_inputs)
            conv3d_inputs = torch.cat([conv3d_inputs[0][0], conv3d_inputs[0][1]], dim=2).unsqueeze(dim=0)

            optical_inputs = self.resizeInputforconv3D(optical_inputs)
            optical_inputs = torch.cat([optical_inputs[0][0], optical_inputs[0][1]], dim=2).unsqueeze(dim=0)

            
            #conv3d
            self.cnn3d.zero_grad()
            cnn3d_out = self.cnn3d(conv3d_inputs)
           
            #opticalconv3d
            self.opticalcnn3d.zero_grad()
            optical_out = self.opticalcnn3d(optical_inputs)

            #twostream
            self.twostream.zero_grad()
            outputs = self.twostream(cnn3d_out, optical_out)
            
            # loss, backpropagation
            l = self.loss(outputs, targets)
            l.backward()
            self.optimizer.step()
            running_loss += l.item()
            total += targets.size(0)
        return total, running_loss


    def test(self, testset, testsetoptical):
        self.cnn3d.eval()
        self.opticalcnn3d.eval()
        self.twostream.eval()
        correct = 0
        total = 0
        with torch.no_grad():
           for conv3d, optical in zip(testset, testsetoptical):
                conv3d_inputs = conv3d[0].to(self.device)
                optical_inputs= optical[0].to(self.device)
                targets = conv3d[1].to(self.device)
                
                conv3d_inputs = self.resizeInputforconv3D(conv3d_inputs)
                conv3d_inputs = torch.cat([conv3d_inputs[0][0], conv3d_inputs[0][1]], dim=2).unsqueeze(dim=0)

                optical_inputs = self.resizeInputforconv3D(optical_inputs)
                optical_inputs = torch.cat([optical_inputs[0][0], optical_inputs[0][1]], dim=2).unsqueeze(dim=0)

                
                #conv3d
                self.cnn3d.zero_grad()
                cnn3d_out = self.cnn3d(conv3d_inputs)
            
                #opticalconv3d
                self.opticalcnn3d.zero_grad()
                optical_out = self.opticalcnn3d(optical_inputs)

                #twostream
                self.twostream.zero_grad()
                outputs = self.twostream(cnn3d_out, optical_out)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return total, correct

    def resizeInputforconv3D(self, inputs):
        batch = inputs.shape[0]
        views = inputs.shape[1]
        frame = inputs.shape[2]
        channel = inputs.shape[3]
        height = inputs.shape[4]
        width = inputs.shape[5]
        inputs = inputs.view(batch, views, channel, frame, height, width)
        return inputs
