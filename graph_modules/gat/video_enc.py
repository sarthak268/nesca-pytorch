import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, max_len):
        super(VideoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        
        # LSTM encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # FC
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, video):
        self.device = video.device

        # Padding the video
        max_x = self.max_len

        if video.size(1) < max_x:
            padding = torch.zeros(video.size(0), max_x - video.size(1), video.size(2)).to(self.device)
            video = torch.cat((video, padding), dim=1)
        
        # LSTM encoding
        hidden = self.init_hidden(video.size(0))
        encoded_video, _ = self.lstm(video, hidden)
        
        # Encoding
        output = self.fc(encoded_video[:, -1, :])
        
        return output
    
    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        return hidden, cell