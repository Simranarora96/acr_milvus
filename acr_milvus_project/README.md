# 🎬 ACR Project with Milvus – Vector-Based Video Fingerprinting

This project is a scalable **Automatic Content Recognition (ACR)** system using **vector embeddings** and **Milvus** for fast and accurate video matching.

Instead of perceptual hashing, it leverages deep learning (ResNet50) to extract frame-level embeddings and uses **Milvus** to store and search these vectors efficiently.

---

## 📁 Project Structure

```plaintext
acr_milvus_project/
├── video_db/               # Folder for known reference videos
├── query_clips/            # Folder for test query clips
├── embeddings/             # Stores extracted embeddings (pkl)
│
├── extract_embeddings.py   # Extracts ResNet50 embeddings from video frames
├── milvus_setup.py         # Creates Milvus collection and schema
├── milvus_insert.py        # Inserts vector embeddings into Milvus
├── milvus_search.py        # Matches query video clip against DB
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install opencv-python torch torchvision pymilvus tqdm
```

### 2. Start Milvus (via Docker)

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.3.3 \
  /milvus/tools/run.sh standalone
```

---

## 🔧 Step-by-Step Usage

### ✅ Step 1: Extract Embeddings from Known Videos

```bash
python extract_embeddings.py
```

### ✅ Step 2: Create Collection in Milvus

```bash
python milvus_setup.py
```

### ✅ Step 3: Insert Video Embeddings

```bash
python milvus_insert.py
```

### ✅ Step 4: Match a Query Clip

1. Add your clip to `query_clips/`
2. Update the path in `milvus_search.py`
3. Run:

```bash
python milvus_search.py
```

---

## 🧠 How It Works

- Uses **ResNet50** to extract a `2048-d` embedding per frame
- Embeddings are stored with metadata (`video_name`) in Milvus
- Query video embeddings are compared to DB embeddings using **L2 distance**
- Frame-level results are aggregated to determine best match

---

## 📌 Notes

- Current frame sampling rate = 1 frame/sec
- Matching based on top-1 voting count
- Easily extendable with CLIP, MobileNet, or scene detection

---

## 🧩 To-Do / Improvements

- [ ] Add Streamlit UI
- [ ] Use scene change detection instead of fixed FPS
- [ ] Support batch video upload + automatic DB population
- [ ] Combine with audio fingerprinting for hybrid ACR



---

## 📊 Why Milvus?

Here's why Milvus is a great fit when you're scaling your ACR system:

| **Feature**         | **Without Milvus**             | **With Milvus**                          |
|---------------------|--------------------------------|-------------------------------------------|
| Number of videos    | < 100 (small-scale)            | 1000+ (scalable)                          |
| Fingerprint type    | Perceptual hash (bit string)   | Deep frame embeddings (vectors)           |
| Matching            | Brute-force (slow)             | Approximate Nearest Neighbors (fast)      |
| Similarity measure  | Hamming distance               | Cosine or Euclidean distance              |
| Query time          | Seconds to minutes             | Milliseconds to seconds                   |

Milvus makes it easy to scale and retrieve similar content across massive datasets with high performance.

