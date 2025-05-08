import torch
from data import ImageDataset
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from piq import FID
import torch
import torch.nn.functional as F

def preprocess(images):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * 0.5 + 0.5
    images = (images - mean) / std
    return images.float()


def calculate_fid(real_loader, generated_loader, device='cuda'):
    # Load pretrained Inception model
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # Modify the model to output features from the pre-logits layer
    model.to(device)
    model.eval()

    # Function to get features from a dataloader
    def get_features(loader):
        features = []
        for images in loader:
            images = preprocess(images)
            with torch.no_grad():
                images = images.to(device)
                output = model(images)
                features.append(output.cpu())
        features = torch.cat(features, dim=0)
        return features

    # Extract features
    real_features = get_features(real_loader)
    gen_features = get_features(generated_loader)

    # Compute FID score
    fid = FID()
    score = fid.compute_metric(real_features, gen_features)
    return score

if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++
    # Please change the path to your dataset
    real_dataset = ImageDataset(root='dataset/test-ffhq', resolution=256, start_id=0, end_id=100)
    fake_dataset = ImageDataset(root='results/pixel/ffhq/inpainting/samples', resolution=256, start_id=0, end_id=100)

    real_loader = DataLoader(real_dataset, batch_size=100, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)

    fid_score = calculate_fid(real_loader, fake_loader)
    print(f'FID Score: {fid_score.item():.4f}')