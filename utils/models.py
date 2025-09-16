import torch
import torch.nn as nn
from tensorflow.keras.models import load_model


class CNN(nn.Module):
    def __init__(self, filter1, filter2, dropout1, dropout2, dropout_fc):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filter1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=filter1, out_channels=filter2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(filter2 * 61, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.batch_norm1 = nn.BatchNorm1d(filter1)
        self.batch_norm2 = nn.BatchNorm1d(filter2)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(-1, self.fc1.in_features)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

def default_peak_finding_model():
    filter1 = 128
    filter2 = 32
    dropout1 = 0.5
    dropout2 = 0.3
    dropout_fc = 0.1
    # Load the pre-trained model
    peak_finding_model = CNN(filter1, filter2, dropout1, dropout2, dropout_fc)
    model_loader = torch.load('utils/models/waveI_cnn.pth')
    peak_finding_model.load_state_dict(model_loader)
    peak_finding_model.eval()
    return peak_finding_model

def default_thresholding_model():
    thresholding_model = load_model('utils/models/abr_thresholding.keras')
    thresholding_model.steps_per_execution = 1
    return thresholding_model