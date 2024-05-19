depth = 64

class Discriminators(nn.Module):
    def __init__(self, numGPU):
        super(Discriminators, self).__init__()
        self.ngpu = numGPU

        self.layer1 = nn.Sequential(spectral_norm(
            nn.Conv2d(in_channels=1, out_channels=depth, kernel_size=(4,4),
            stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(spectral_norm(
            nn.Conv2d(in_channels=depth, out_channels=depth*2, kernel_size=(4,4),
            stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(depth*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(spectral_norm(
            nn.Conv2d(in_channels=depth*2, out_channels=depth*4, kernel_size=(4,4),
            stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(depth*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(spectral_norm(
            nn.Conv2d(in_channels=depth*4, out_channels=depth*8, kernel_size=(3,3),
            stride=3, padding=1, bias=False)),
            nn.BatchNorm2d(depth*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(spectral_norm(
            nn.Conv2d(in_channels=depth*8, out_channels=depth*16, kernel_size=(3,3),
            stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(depth*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer6 = nn.Sequential(spectral_norm(
            nn.Conv2d(in_channels=depth*16, out_channels=depth*32, kernel_size=(3,3),
            stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(depth*32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.outputLayer = nn.Sequential(
            nn.Conv2d(in_channels=depth*32, out_channels=1,
            kernel_size=(3,3), stride=1, padding=0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, inputImage):
        layer1 = self.layer1(inputImage)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        return self.outputLayer(layer6)
