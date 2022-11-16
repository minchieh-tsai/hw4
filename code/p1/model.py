
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


if __name__ == '__main__':

	inputs = torch.zeros(8, 3, 84, 84)
	# inputs = inputs.to(device)
	conv4 = Convnet()
	# vae = vae.to(device)
	output = conv4(inputs) #torch.Size([8, 1600])