import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.utils_rotnist import save_dataset, load_saved_dataset
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

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=32):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
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
    
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class AE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=32):
        super(AE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        z = self.fc2(h1)
        return z

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

def ae_loss_function(recon_x, x):
  BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
  return BCE

def vae_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def ae_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
#       data = data.view(data.size(0), -1).to(device)  # Flatten the data
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = ae_loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def encode_data(model, data_loader, model_type, device):
    model.eval()
    latent_array = []
    original_images_paths = []
    with torch.no_grad():
        for data, paths in data_loader:
            data = data.to(device)
            if model_type == 'vae':
                mu, logvar = model.encode(data)
                z = model.reparameterize(mu, logvar)
            elif model_type == 'ae':
                z = model.encode(data)
            latent_array.append(z)
            original_images_paths.extend(paths)
    concat_latent_array = torch.cat(latent_array)
    print(f"concat latent array shape: {concat_latent_array.shape}")
    return concat_latent_array, original_images_paths

def decode_data(model, z, device):
    model.eval()
    with torch.no_grad():
        z = z.to(device)
        decoded = model.decode(z)
    return decoded

def run_encoding(model, device, train_loader, model_type, encoded_dir):
    # Save encoded data
    latent_array, original_images_paths = encode_data(model, train_loader, model_type, device)
    os.makedirs(encoded_dir, exist_ok=True)

    set_index = 1
    for i in range(latent_array.size(0)):
        set_image_index = (i % T) + 1
        torch.save((latent_array[i], original_images_paths[i]), os.path.join(encoded_dir, f'enc_img_{set_index}_{set_image_index}.pt'))
        if set_image_index == T:
            set_index += 1

def run_decoding(model, device, pkl_path, latent_dim, T, smnr_db):
    pkl_file = sorted(glob.glob(os.path.join(pkl_path, f"sequence_m_{latent_dim}_n_{latent_dim}_rotnist_T_{T}_*_smnr_{smnr_db}dB.pkl")))
    Z_XY_dict = load_saved_dataset(str(pkl_file[0]))
    y_array = Z_XY_dict["dataY"]
    z_array = Z_XY_dict["dataZ"]
    fp_arr = Z_XY_dict["img_fpaths"]
    decoded_z_list = list()
    decoded_y_list = list()
    reqd_fpaths = list()
    cnt = 0
    for noise_vector in y_array:
        decoded_y = decode_data(model, torch.tensor(noise_vector[0]), device)
        decoded_y_list.append(decoded_y)
        cnt += 1
        if cnt == 5:
            cnt = 0
            break
    
    for latent_vector in z_array:
        decoded_z = decode_data(model, torch.tensor(latent_vector[0]), device)
        decoded_z_list.append(decoded_z)
        cnt += 1
        if cnt == 5:
            cnt = 0
            break

    for fpath in fp_arr:
        reqd_fpaths.append(fpath[0])
        cnt += 1
        if cnt == 5:
            break
    
    return decoded_y_list, decoded_z_list, reqd_fpaths

def generate_danse_input(encoded_dir, output_dir, smnr_db=10):
    os.makedirs(output_dir, exist_ok=True)
    Z_XY = dict()
    Z_arr = list()
    Y_arr = list()
    Cw_arr = list()
    fp_arr = list()
    # encoded_images = []
    # for set_index in range(1, (len(glob.glob(os.path.join(encoded_dir, '*.pt'))) // T) + 1):
    #     for set_image_index in range(1, T + 1):
    #         encoded_images.append(torch.load(os.path.join(encoded_dir, f'enc_img_{set_index}_{set_image_index}.pt')))
    encoded_images = []
    og_img_fpaths = []
    pt_files = sorted(glob.glob(os.path.join(encoded_dir, '*.pt')))
    
    for file_path in pt_files:
        data = torch.load(file_path)
        encoded_img = data[0]
        og_img_fp = data[1]
        encoded_images.append(encoded_img)
        og_img_fpaths.append(og_img_fp)
        
        # Check if we have collected T encoded images
        if len(encoded_images) == T:
            Z_arr.append(encoded_images)
            fp_arr.append(og_img_fpaths)
            encoded_images = []
            og_img_fpaths = []

    # Check if there are any remaining images that were not added
    if encoded_images:
        Z_arr.append(encoded_images)

    for z_vector in Z_arr:
        z_tensor = torch.stack(z_vector)
        
        # Calculate signal power using variance
        signal_power = torch.var(z_tensor.cpu())
        
        # Calculate noise power from SMNR in dB
        noise_power = signal_power / (10**(smnr_db / 10))
        
        # Generate Gaussian noise with the calculated noise power
        noise_std = torch.sqrt(noise_power)
        noise = noise_std * torch.randn_like(z_tensor[0])  # Generate noise for one element
        
        # Add the same noise to each element of the z_vector
        y_vector = [latent_vector + noise for latent_vector in z_tensor]
        
        # Append to Y_arr and add the variance to Cw_arr
        Y_arr.append(y_vector)
        #Cw_arr.append(noise_power.item())
        Cw_matrix = noise_power.item()*np.eye(latent_dim)
        Cw_arr.append(Cw_matrix)
    
    num_samples = len(Z_arr)
    Z_XY["dataZ"] = np.asarray([[latent_vector.cpu().numpy() for latent_vector in z_list] for z_list in Z_arr])
    Z_XY["dataY"] = np.asarray([[latent_vector.cpu().numpy() for latent_vector in decoded_list] for decoded_list in Y_arr])
    Z_XY["dataCw"] = np.asarray(Cw_arr)
    Z_XY["img_fpaths"] = fp_arr

    filename = f"sequence_m_{latent_dim}_n_{latent_dim}_rotnist_T_{T}_N_{num_samples}_smnr_{smnr_db}dB.pkl"
    savepath = os.path.join(output_dir, filename)
    save_dataset(Z_XY=Z_XY, filename=savepath)
    print(f"Saved .pkl file to: {savepath}")
    

# Save reconstructed images to PDF
def save_reconstructed_images(decoded_y_list, decoded_z_list, reqd_fpaths, filename='reconstructed_images.pdf'):
    with torch.no_grad():
        with PdfPages(filename) as pdf:
            for i in range(len(decoded_y_list)):  # Save only 5 images
                original_image = Image.open(reqd_fpaths[i]).convert('L')
                original_image = transforms.ToTensor()(original_image).view(28, 28)
                plt.figure(figsize=(8, 4))
                # Original Image
                plt.subplot(1, 3, 1)
                plt.imshow(original_image, cmap='gray')
                plt.title('Original')
                plt.axis('off')
                # Reconstructed Latent Image
                plt.subplot(1, 3, 2)
                plt.imshow(decoded_z_list[i].cpu().view(28, 28), cmap='gray')
                plt.title('Reconstructed Latent')
                plt.axis('off')
                # Reconstructed Noised Image
                plt.subplot(1, 3, 3)
                plt.imshow(decoded_y_list[i].cpu().view(28, 28), cmap='gray')
                plt.title('Reconstructed Noised')
                plt.axis('off')
                pdf.savefig()
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for ROTNIST")
    parser.add_argument("--mode", help="Enter 'encode' or 'decode' mode", type=str, default="train")
    parser.add_argument("--model_type", help="Enter 'ae' or 'vae' model", type=str, default="ae")
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default='data/encoded_data/')
    parser.add_argument("--danse_input_path", help="Enter full path to store the danse input files", type=str, default='data/encoded_noise_data')
    parser.add_argument("--saved_model_path", help="Enter full path to save the AR/VAE model", type=str, default='models/rotnist_models/')
    parser.add_argument("--smnr_db", help="For the smnr", type=float, default=20.0)
    parser.add_argument("--latent_dim", help="For the latent dimension", type=int, default=32)


    args = parser.parse_args() 
    mode = args.mode
    model_type = args.model_type
    enc_img_output_dir = args.output_path
    saved_model_path = args.saved_model_path
    danse_input_path = args.danse_input_path
    smnr_db = args.smnr_db
    latent_dim = args.latent_dim
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type.lower() == 'vae':
        model = VAE(latent_dim=latent_dim).to(device)
        train = vae_train
    elif model_type.lower() == 'ae':
        model = AE(latent_dim=latent_dim).to(device)
        train = ae_train

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    if mode.lower() == 'train':
        # Training loop
        num_epochs = 10  # Increase the number of epochs
        for epoch in range(1, num_epochs + 1):
            train_loss = train(model, device, train_loader, optimizer, epoch)
            print(f'Epoch {epoch}, Loss: {train_loss}')
        os.makedirs(saved_model_path, exist_ok=True)
        torch.save(model, os.path.join(saved_model_path, f"{model_type}_model"))
        print(f"Saved model to: {saved_model_path}")

    elif mode.lower() == 'encode':
        model = torch.load(os.path.join(saved_model_path, f"{model_type}_model"))
        run_encoding(model, device, train_loader, model_type, enc_img_output_dir)
        print(f"Saved encodings to: {enc_img_output_dir}")
        
    elif mode.lower() == 'noise':
        print("Adding noise for {}dB SMNR".format(smnr_db))
        generate_danse_input(enc_img_output_dir, danse_input_path, smnr_db=smnr_db)
        print(f"Saved noisy encodings to: {danse_input_path}")
    
    # Change it to decode the processed danse output
    elif mode.lower() == 'decode':
        model = torch.load(os.path.join(saved_model_path, f"{model_type}_model"))
        decoded_y_list, decoded_z_list, reqd_fpaths = run_decoding(model, device, danse_input_path, latent_dim, T, smnr_db)
        save_reconstructed_images(decoded_y_list, decoded_z_list, reqd_fpaths)
        print("Decoded from latent space succesfully. Check plots!")

    else:
        print("Invalid mode. Please select 'train', 'encode', or 'decode'.")
