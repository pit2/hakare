import torch.nn
from torchvision import transforms
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Generator(torch.nn.Module):
    def __init__(self, input_size, features, channels):
        super().__init__()
        self.gen = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=input_size, out_channels=features*8, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(features*8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(in_channels=features*8, out_channels=features*4, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.BatchNorm2d(features*4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(in_channels=features*4, out_channels=features*2, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.BatchNorm2d(features*2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(in_channels=features*2, out_channels=features, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(in_channels=features, out_channels=channels, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(torch.nn.Module):
    def __init__(self, features, channels):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=4,
                            stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(features*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=4,
                            stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(features*4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=4,
                            stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(features*8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # output as 0 or 1
            torch.nn.Conv2d(features*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.main(inputs)


BATCH_SIZE = 128
transform = transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
mnist = torchvision.datasets.MNIST('./var', download=True)
real = torchvision.datasets.MNIST('./var', train=True, transform=transform)
real_dl = torch.utils.data.DataLoader(
    real, batch_size=BATCH_SIZE, shuffle=True)

epochs = 16
# Real or Fake
real_label = 1
fake_label = 0

context_size = 10
features = 32
channels = 1


# Binary Loss (real or fake)
criterion = torch.nn.BCELoss()

# Random Number Generator
fixed_noise = torch.randn(features, context_size, 1, 1, device=DEVICE)

# List to Track Progress
img_list = []
gen_losses = []
disc_losses = []

discriminator = Discriminator(features, channels).to(DEVICE)
generator = Generator(context_size, features, channels).to(DEVICE)
# Adam Optmiziers
optim_disc = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))
optim_gen = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))


fake = generator(fixed_noise).detach().cpu()
samples = torchvision.utils.make_grid(fake, padding=2, normalize=True)
plt.axes().imshow(samples.permute(1, 2, 0))

for epoch in range(epochs):
    with tqdm(real_dl, unit='batches') as progress:
        for data, _ in real_dl:
            # for i , (data , _) in enumerate(realloader):

            # Preparing Data
            batch_size = data.shape[0]
            real_data = data.to(DEVICE)
            real_labels = torch.full((batch_size,), 1.0, device=DEVICE)
            discriminator.zero_grad()
            output = discriminator(real_data).view(-1)
            # Loss Function for Real or Fake
            disc_err_on_real = criterion(output, real_labels)
            disc_err_on_real.backward()
            # Creating Noise and Fake Labels
            noise = torch.randn(batch_size, context_size, 1, 1, device=DEVICE)
            fake_labels = torch.full((batch_size,), 0.0, device=DEVICE)
            fake_data = generator(noise)
            output = discriminator(fake_data).view(-1)
            disc_err_on_fake = criterion(output, fake_labels)
            disc_err_on_fake.backward(retain_graph=True)
            total_disc_err = disc_err_on_real + disc_err_on_fake
            optim_disc.step()

            # How well is Discriminator is working
            fake_labels.fill_(real_label)
            generator.zero_grad()
            output = discriminator(fake_data).view(-1)
            gen_err = criterion(output, real_labels)
            gen_err.backward()
            # Update Optimizer
            optim_gen.step()
            # Saving Losses
            gen_losses.append(gen_err.item())
            disc_losses.append(total_disc_err.item())
            progress.set_postfix(gen_loss=torch.tensor(gen_losses).mean(),
                                disc_loss=torch.tensor(disc_losses).mean(),
                                refresh=False)
            progress.update()
            gen_losses.append(gen_err.item())
            disc_losses.append(total_disc_err.item())
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        samples = torchvision.utils.make_grid(fake, padding=2, normalize=True)
        img_list.append(samples)
