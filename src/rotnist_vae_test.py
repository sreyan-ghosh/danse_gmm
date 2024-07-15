import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class RotatedMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=9):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Save reconstructed images to PDF
def save_reconstructed_images(model, data_loader, device, filename='reconstructed_images.pdf'):
    model.eval()
    with torch.no_grad():
        with PdfPages(filename) as pdf:
            for batch_idx, data in enumerate(data_loader):
                data = data.to(device)
                recon_batch, _, _ = model(data)
                for i in range(min(len(recon_batch), 5)):  # Save only 5 images for brevity
                    plt.figure(figsize=(4, 2))
                    # Original Image
                    plt.subplot(1, 2, 1)
                    plt.imshow(data[i].cpu().view(28, 28), cmap='gray')
                    plt.title('Original')
                    plt.axis('off')
                    # Reconstructed Image
                    plt.subplot(1, 2, 2)
                    plt.imshow(recon_batch[i].cpu().view(28, 28), cmap='gray')
                    plt.title('Reconstructed')
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),   # Convert to tensor
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the tensor
    ])

    # Load dataset
    train_dataset = RotatedMNISTDataset(root_dir='data/rotnist/train-images/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Training loop
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        
    save_reconstructed_images(model, train_loader, device)

if __name__ == "__main__":
    main()
