import numpy as np
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, data):
        pass
    

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        logits = self.out(x)
        return logits
      
        
class CNNModel(DigitClassificationInterface):
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = CNN()
            
    def predict(self, image):
        data = image.permute(2, 0, 1).unsqueeze(0).float()
        
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(data)
        
        predicted_class_idx = logits.argmax(-1).item()
        
        return predicted_class_idx
    
    
class RandomForestModel(DigitClassificationInterface):
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = RandomForestClassifier()
            random_images = np.random.rand(100,784)
            random_output = np.random.randint(0, 10, size=100)
            self.model.fit(random_images,random_output)
            
    def predict(self, image):
        data = image.reshape(1, -1)
        return self.model.predict(data)[0]
    
    
class RandomModel(DigitClassificationInterface):
    def predict(self, image):
        return np.random.randint(0, 10)
    
    
class DigitClassifier:
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.model = CNNModel()
            self.convert_input = torch.from_numpy
        elif algorithm == 'rf':
            self.model = RandomForestModel()
            self.convert_input = lambda image: image.flatten()
        elif algorithm == 'rand':
            self.model = RandomModel()
            self.convert_input = lambda image: image[14-5:14+5,14-5:14+5,0]
        else:
            raise ValueError("Unknown algorithm specified")
        
    def check_input(self, image):
        if not isinstance(image, np.ndarray) or image.shape != (28, 28, 1):
            raise ValueError("Input must be a 28x28x1 numpy array.")

    def predict(self, image):
        self.check_input(image)
        image = self.convert_input(image)
        return self.model.predict(image)
    
    def train(self):
         raise NotImplementedError("Training not implemented.")
    
    
if __name__ == '__main__':
    image = (np.random.rand(28,28,1)* 255).astype(np.uint8)
    
    dc = DigitClassifier('rf')
    
    pred = dc.predict(image)
    print(pred)