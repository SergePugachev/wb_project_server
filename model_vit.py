import torch
import torch.nn.functional as f
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision
from torch import nn, optim

# Трансформации для изображений
transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1).to(device)
for param in model_vit.parameters():
    param.requires_grad = False

model_vit.heads = nn.Linear(in_features=768, out_features=2).to(device)

criterion_vit = nn.CrossEntropyLoss()
optimizer_vit = optim.Adam(model_vit.parameters(), lr=0.0003)
model_vit.load_state_dict(torch.load('vit_weights.pt', map_location=torch.device('cpu')))


def result_vit(image):
    """
    Функция принимает на вход изображение, конвертирует его, затем пропускает через список трансформаций,
    классифицирует его и возвращает результаты в виде вероятностей класса
    """
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    image = transforms(image)
    image = torch.unsqueeze(image, 0)
    model_vit.eval()
    pred_logits_tensor = model_vit(image)
    pred_probs = f.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    normal, fraud = pred_probs[0][0], pred_probs[0][1]
    return normal, fraud
