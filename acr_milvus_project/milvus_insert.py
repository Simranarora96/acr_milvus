import os
import pickle
from pymilvus import Collection
import numpy as np

collection = Collection("video_fingerprints")

def insert_video_embeddings(video_name, embeddings):
    video_names = [video_name] * len(embeddings)
    data = [video_names, embeddings.tolist()]
    collection.insert(data)
    print(f"✅ Inserted {len(embeddings)} vectors for '{video_name}'")

if __name__ == "__main__":
    embed_dir = "embeddings"
    for fname in os.listdir(embed_dir):
        if fname.endswith(".pkl"):
            video_name = os.path.splitext(fname)[0]
            with open(os.path.join(embed_dir, fname), "rb") as f:
                embs = pickle.load(f)
            insert_video_embeddings(video_name, embs)
