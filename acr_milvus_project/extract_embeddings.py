import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet50
import torchvision.transforms as transforms
import pickle

# Load ResNet50 (without final classifier)
model = resnet50(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_frame_embeddings(video_path, every_n_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * every_n_sec)
    frame_idx = 0
    embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                embedding = model(tensor).squeeze().numpy()
            embeddings.append(embedding)
        frame_idx += 1
    cap.release()
    return np.array(embeddings)

if __name__ == "__main__":
    input_dir = "video_db"
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir), desc="Extracting embeddings"):
        if fname.endswith((".mp4", ".mov", ".mkv")):
            path = os.path.join(input_dir, fname)
            embs = extract_frame_embeddings(path)
            out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(embs, f)
