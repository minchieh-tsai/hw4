
import torch
import torch.nn as nn

class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # in: 3 x 84 x 84

            conv_block(x_dim, hid_dim),
            # out: 64 x 42 x 42

            conv_block(hid_dim, hid_dim),
            # out: 64 x 21 x 21

            conv_block(hid_dim, hid_dim),
            # out: 64 x 10 x 10

            conv_block(hid_dim, z_dim),
            # out: 64 x 5 x 5 -> 1600
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Hallucinator(nn.Module):

    def __init__(self):
        super().__init__()
        bn1 = nn.BatchNorm1d(1024)
        nn.init.uniform_(bn1.weight)
        bn2 = nn.BatchNorm1d(1600)
        nn.init.uniform_(bn2.weight)
        self.generator = nn.Sequential(
            # in: 1600

            nn.Linear(1600, 1024),
            bn1,
            nn.ReLU(True),
            nn.Dropout2d(),
            # out: 1024

            nn.Linear(1024, 1600),
            bn2,
            nn.ReLU(True),
            # out: 1600
        )

    def forward(self, x):
        x = self.generator(x)
        return x

if __name__ == '__main__':
    batch = 8
    device = 'cpu'
    
    inputs = torch.zeros(batch, 3, 84, 84)
    # inputs = inputs.to(device)
    conv4 = Convnet()
    # vae = vae.to(device)
    output = conv4(inputs) #torch.Size([8, 1600])
    print(output.size())
    
    
    hallu = Hallucinator()
    noise = torch.randn(batch, 1600, device=device)
    fake = hallu(output + noise)
    print(fake.size())
    
    
    
    