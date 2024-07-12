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
import argparse

class RotatedMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
        self.set_counts = self.count_sets()
        self.T = self.set_counts['1']
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image, img_path

    def count_sets(self):
        set_counts = {}
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            set_number = filename.split('_')[0]
            if set_number not in set_counts:
                set_counts[set_number] = 0
            set_counts[set_number] += 1
        return set_counts

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),   # Convert to tensor
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the tensor
])

# Load dataset
train_dataset = RotatedMNISTDataset(root_dir='data/rotnist/train-images/', transform=transform)
T = train_dataset.T
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
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
"""
# Training loop
for epoch in range(1, 11):
    train(epoch)
"""
def encode_data(model, data_loader, device):
    model.eval()
    encoded_mu = []
    encoded_logvar = []
    original_images_paths = []
    with torch.no_grad():
        for data, paths in data_loader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            encoded_mu.append(mu)
            encoded_logvar.append(logvar)
            original_images_paths.extend(paths)
    return torch.cat(encoded_mu), torch.cat(encoded_logvar), original_images_paths

def decode_data(model, z, device):
    model.eval()
    with torch.no_grad():
        z = z.to(device)
        decoded = model.decode(z)
    return decoded

def run_encoding(encoded_dir='data/encoded_data/'):
    # Save encoded data
    encoded_mu, encoded_logvar, original_images_paths = encode_data(model, train_loader, device)
    os.makedirs(encoded_dir, exist_ok=True)

    set_index = 1
    for i in range(encoded_mu.size(0)):
        set_image_index = (i % T) + 1
        torch.save((encoded_mu[i], encoded_logvar[i], original_images_paths[i]), os.path.join(encoded_dir, f'enc_img_{set_index}_{set_image_index}.pt'))
        if set_image_index == T:
            set_index += 1

def run_decoding(encoded_dir):
    # Load encoded data with new name format
    encoded_images = []
    for set_index in range(1, (len(glob.glob(os.path.join(encoded_dir, '*.pt'))) // T) + 1):
        for set_image_index in range(1, T + 1):
            encoded_images.append(torch.load(os.path.join(encoded_dir, f'enc_img_{set_index}_{set_image_index}.pt')))

    # Load encoded data and decode
    encoded_mu_loaded = torch.stack([img[0] for img in encoded_images])
    encoded_logvar_loaded = torch.stack([img[1] for img in encoded_images])
    original_images_paths_loaded = [img[2] for img in encoded_images]

    # Reparameterize
    z = model.reparameterize(encoded_mu_loaded, encoded_logvar_loaded)

    # Decode
    decoded_images = decode_data(model, z, device)
    
    return decoded_images, original_images_paths_loaded

# Save reconstructed images to PDF
def save_reconstructed_images(original_images_paths, decoded_images, filename='reconstructed_images_enc_dec.pdf'):
    with PdfPages(filename) as pdf:
        for i in range(min(len(decoded_images), 5)):  # Save only 5 images
            original_image = Image.open(original_images_paths[i]).convert('L')
            original_image = transforms.ToTensor()(original_image).view(28, 28)
            plt.figure(figsize=(8, 4))
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            # Reconstructed Image
            plt.subplot(1, 2, 2)
            plt.imshow(decoded_images[i].cpu().view(28, 28), cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')
            pdf.savefig()
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for ROTNIST")
    parser.add_argument("--mode", help="Enter 'encode' or 'decode' mode", type=str, default="encode")
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default='data/encoded_data/')

    args = parser.parse_args() 
    mode = args.mode
    output_path = args.output_path
    
    if mode.lower() == 'encode':
        # Training loop
        for epoch in range(1, 11):
            train(epoch)
        run_encoding(output_path)
    
    elif mode.lower() == 'decode':
        decoded_images, original_image_path_loaded = run_decoding(output_path)
        save_reconstructed_images(original_image_path_loaded, decoded_images)
