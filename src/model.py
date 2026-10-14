"""
PyTorch Deep Neural Network architecture for Hand Gesture Classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureClassifierNN(nn.Module):
    """
    Deep Neural Network for robust, real-time hand gesture recognition.
    Takes normalized 3D hand landmarks and invariant geometric features.
    """
    def __init__(self, input_dim: int = 73, num_classes: int = 8, dropout_rate: float = 0.2):
        super(GestureClassifierNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.out = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single sample during inference (shape: [input_dim] -> [1, input_dim])
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        
        logits = self.out(x)
        return logits

    def predict(self, x: torch.Tensor):
        """
        Inference helper returning the predicted class index and confidence probability.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            return predicted_idx.item(), confidence.item(), probabilities.squeeze(0).cpu().numpy()
