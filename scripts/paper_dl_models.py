import torch
import torch.nn as nn

class Paper_FFNN(nn.Module):
    """
    Feed-Forward Neural Network as described by Pancotti et al. (2022).
    Input: Static / Flattened Longitudinal Features only.
    Layout: 3 Hidden Layers (64, 32, 16) with ReLU and 0.3 Dropout.
    Output: 1 Linear Node for regression.
    """
    def __init__(self, num_static_features):
        super(Paper_FFNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(num_static_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(16, 1) # Linear activation output
        )

    def forward(self, static_x, dynamic_x=None):
        # We accept dynamic_x just so the training loop can pass both
        # inputs consistently to all models without breaking
        return self.network(static_x)


class Paper_CNN_Hybrid(nn.Module):
    """
    CNN + FFNN as described by the base paper.
    Dynamic Input: (Batch, Channels=1, 11 Questions, 5 Visits)
    Layout: 
      Conv2d(11x3) -> Conv2d(1x3) -> Flatten
      Concat w/ Static Input -> FFNN
    """
    def __init__(self, num_static_features, num_cnn_filters=8):
        super(Paper_CNN_Hybrid, self).__init__()
        
        # Filter 1: 11x3, out height becomes 11-11+1 = 1
        # Width becomes 5-3+1 = 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_cnn_filters, kernel_size=(11, 3))
        self.relu1 = nn.ReLU()
        
        # Filter 2: 1x3, applied to the (1 x Width=3) output
        # Width becomes 3-3+1 = 1
        self.conv2 = nn.Conv2d(in_channels=num_cnn_filters, out_channels=num_cnn_filters, kernel_size=(1, 3))
        self.relu2 = nn.ReLU()
        
        # Final CNN feature vector size = num_cnn_filters * 1 height * 1 width
        self.cnn_flat_size = num_cnn_filters
        
        # Merged FFNN Stream
        self.ffnn = nn.Sequential(
            nn.Linear(num_static_features + self.cnn_flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, static_x, dynamic_x):
        # dynamic_x shape: (Batch, 1 Channel, 11 Features, 5 Visits)
        cnn_out = self.relu1(self.conv1(dynamic_x))
        cnn_out = self.relu2(self.conv2(cnn_out))
        
        # Flatten CNN output
        cnn_out = cnn_out.view(cnn_out.size(0), -1) 
        
        # Concatenate CNN features with static features
        merged = torch.cat((static_x, cnn_out), dim=1)
        
        # Feed into FFNN
        return self.ffnn(merged)


class Paper_RNN_Hybrid(nn.Module):
    """
    RNN (LSTM) + FFNN as described by the base paper.
    Dynamic Input: (Batch, Channels=1, 11 Questions, 5 Visits)
    Layout:
      LSTM(Input=11, Hidden=16, Layers=2) -> Concat Last State w/ Static -> FFNN
    """
    def __init__(self, num_static_features, rnn_hidden_size=16):
        super(Paper_RNN_Hybrid, self).__init__()
        
        # Input to LSTM should be (Batch, Seq_Len=5 visits, Features=11 questions)
        self.rnn = nn.LSTM(input_size=11, hidden_size=rnn_hidden_size, num_layers=2, batch_first=True)
        
        # Merged FFNN Stream
        self.ffnn = nn.Sequential(
            nn.Linear(num_static_features + rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, static_x, dynamic_x):
        # Remove the channel dimension and transpose 
        # From (Batch, 1, 11 Features, 5 Visits) --> (Batch, 5 Visits, 11 Features)
        dynamic_x = dynamic_x.squeeze(1).transpose(1, 2)
        
        # Pass to RNN
        rnn_out, (h_n, c_n) = self.rnn(dynamic_x)
        
        # Get the final hidden state from the last LSTM layer (h_n shape: num_layers, batch, hidden_size)
        final_rnn_state = h_n[-1] 
        
        # Concatenate and pass to FFNN
        merged = torch.cat((static_x, final_rnn_state), dim=1)
        return self.ffnn(merged)
