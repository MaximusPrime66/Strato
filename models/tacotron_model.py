import torch
import torch.nn as nn
import numpy as np

class TacotronModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TacotronModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Embedding(input_dim, 512),
            nn.LSTM(512, 256, batch_first=True)
        )
        
        self.decoder = nn.Sequential(
            nn.LSTM(256, 512, batch_first=True),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, text):
        encoded = self.encoder(text)[0]
        decoded = self.decoder(encoded)[0]
        return decoded
    
    def inference(self, text):
        with torch.no_grad():
            mel_outputs = self.forward(text)
            return mel_outputs, mel_outputs, None, None

def load_model(path):
    model = TacotronModel(input_dim=256, output_dim=80)
    return model