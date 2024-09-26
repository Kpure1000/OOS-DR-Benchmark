import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class AutoEncoderModule(nn.Module):

    def __init__(self, input_dim, hidden_dims, device):
        super(AutoEncoderModule, self).__init__()
        torch.random.manual_seed(0)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device
        
        encoder_layers = []
        decoder_layers = []
        # 创建编码器层
        for i in range(len(self.hidden_dims) - 1):
            if i == 0:
                encoder_layers.append(nn.Linear(self.input_dim, self.hidden_dims[i]))
                encoder_layers.append(nn.Tanh())
                encoder_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                encoder_layers.append(nn.Tanh())
            else:
                encoder_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))

        # 创建解码器层
        for i in range(len(self.hidden_dims) - 1, 0, -1):
            if i - 1 > 0:
                decoder_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1]))
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1]))
                decoder_layers.append(nn.Tanh())
                decoder_layers.append(nn.Linear(self.hidden_dims[i - 1], self.input_dim))
                decoder_layers.append(nn.Sigmoid())

        # pprint(encoder_layers)
        # pprint(decoder_layers)

        # 组合编码器和解码器
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoder:
    
    def __init__(self, input_dim, hidden_dims=[256, 64, 16, 2], device="cpu"):
        self.module = AutoEncoderModule(input_dim, hidden_dims, device)
    
    def fit(self, data, n_epochs=100, lr=1e-3, verbose=False):
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.module.device)
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.module.parameters(), lr=lr)

        if verbose:
            loop = tqdm(range(n_epochs), desc="Epochs", ncols=100, unit="txt")
        else:
            loop = range(n_epochs)

        for epoch in loop:
            _, decoded= self.module(data_tensor)
            
            loss = loss_function(decoded, data_tensor)
            
            optimizer.zero_grad()

            loss.backward()
            
            optimizer.step()
            
            if verbose:
                loop.set_description(f'Epoch [{epoch}/{n_epochs}]')
                loop.set_postfix(loss = f"{loss.item():.4f}")

    def transform(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.module.device)
        encoded_data = self.module.encoder(data_tensor).cpu().detach().numpy()
        return encoded_data
