depth = 32
noise = 100

class Generators(nn.Module):
    def __init__(self, numGPU):
        super(Generators, self).__init__()
        self.ngpu = numGPU

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise, out_channels=depth*32,
            kernel_size=(3,3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=depth*32),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=depth*32, out_channels=depth*16,
            kernel_size=(3,3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depth*16),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=depth*16, out_channels=depth*8,
            kernel_size=(3,3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depth*8),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=depth*8, out_channels=depth*4,
            kernel_size=(3,3), stride=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depth*4),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=depth*4, out_channels=depth*2,
            kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depth*2),
            nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=depth*2, out_channels=depth,
            kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depth),
            nn.ReLU(inplace=True)
        )

        self.outputLayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=depth, out_channels=1,
            kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        layer1 = self.layer1(noise)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        return self.outputLayer(layer6)
