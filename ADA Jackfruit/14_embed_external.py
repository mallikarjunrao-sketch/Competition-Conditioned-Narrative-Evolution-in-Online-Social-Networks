
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk-size", type=int, default=25000)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out_dir)
    chunks = out / "embedding_chunks"
    chunks.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    texts = df.sort_values("text_idx")["text_clean"].tolist()
    n = len(texts)

    print(f"Unique texts: {n:,}", flush=True)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    model = SentenceTransformer(
        args.model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    metadata = {
        "model": args.model,
        "dimension": 384,
        "unique_texts": n,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "normalized": True
    }

    metadata_path = chunks / "metadata.json"

    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old != metadata:
            raise RuntimeError("Embedding configuration differs from existing chunks.")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))

    total_chunks = (n + args.chunk_size - 1) // args.chunk_size

    for i in range(total_chunks):
        start = i * args.chunk_size
        end = min(start + args.chunk_size, n)
        chunk_path = chunks / f"embedding_{i:04d}.npy"

        if chunk_path.exists():
            existing = np.load(chunk_path, mmap_mode="r")
            if existing.shape == (end-start, 384):
                print(f"Skipping completed chunk {i+1}/{total_chunks}")
                continue
            chunk_path.unlink()

        print(
            f"\nEmbedding chunk {i+1}/{total_chunks}: "
            f"texts {start:,}–{end:,}",
            flush=True
        )

        embeddings = model.encode(
            texts[start:end],
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        if embeddings.shape != (end-start, 384):
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

        temp_path = chunk_path.with_suffix(".tmp.npy")
        np.save(temp_path, embeddings)
        temp_path.replace(chunk_path)

        print(f"Saved: {chunk_path}", flush=True)

    print("\n=== EMBEDDING COMPLETE ===")
    print("Chunks:", total_chunks)
    print("Texts embedded:", f"{n:,}")
    print("Output directory:", chunks)

if __name__ == "__main__":
    main()
