import torch
import torch.nn as nn

class KnowledgeWeightingModel(nn.Module):
    '''
    This model takes in an input with variable shape (N, D) where N can take any value with 
    max value as args.vocab_size. This input is padded and then passed through an RNN that 
    weighs each input of dimension D equal by performing mean pooling over it. 
    '''
    
    def __init__(self, args):
        super(KnowledgeWeightingModel, self).__init__()

        self.args = args
        
        latent_dim = 64 if args.dataset == '50salads' else 16
        
        if args.use_gsnn:
            self.max_length = args.vocab_size
            
            if args.rectification_method == 'diagonal' or args.rectification_method == 'weighting':
                hidden_dim = latent_dim // 2
                self.rnn = nn.GRU(input_size=args.context_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_dim*2, latent_dim)  # Additional linear layer for output size adjustment

            elif args.rectification_method == 'full':
                hidden_dim = 1024
                self.rnn = nn.GRU(input_size=args.context_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_dim*2, latent_dim*latent_dim)  # Additional linear layer for output size adjustment
      
        else:
            self.fc = nn.Linear(args.vocab_size * args.state_dim, latent_dim)


    def forward(self, x):
        
        if self.args.use_gsnn:
            batch_size = x.size(0)

            if x.size(1) < self.max_length:
                pad_length = self.max_length - x.size(1)
                padding = torch.zeros(batch_size, pad_length, x.size(2)).to(x.device)
                x = torch.cat((x, padding), dim=1)
            
            elif x.size(1) > self.max_length:
                x = x[:, :self.max_length, :]  # Truncate the sequence if it exceeds the maximum length

            output, _ = self.rnn(x)
            output = torch.mean(output, dim=1)  # Global average pooling
            output = self.fc(output)  # Additional linear layer for output size adjustment

        else:
            output = self.fc(x.view(x.shape[0], -1))

        return output