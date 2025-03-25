import cv2
import torch
import numpy as np
from PIL import Image
from pymilvus import Collection
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Load model
model = resnet50(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding_from_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze().numpy()
    return emb

collection = Collection("video_fingerprints")
collection.load()

def match_query_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 1)
    frame_idx = 0
    results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            query_emb = get_embedding_from_frame(frame)
            search_res = collection.search(
                data=[query_emb],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=1,
                output_fields=["video_name"]
            )
            top_match = search_res[0][0].entity.get("video_name")
            results[top_match] = results.get(top_match, 0) + 1
        frame_idx += 1

    best_match = max(results, key=results.get)
    print(f"🎯 Best match for {video_path}: {best_match}")
    return best_match

if __name__ == "__main__":
    match_query_video("query_clips/sample.mp4")
