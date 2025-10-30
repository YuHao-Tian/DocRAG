import argparse
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def iter_texts(extract_dir: Path):
    for p in extract_dir.glob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                img = ex["image"]
                for ch in ex.get("chunks", []):
                    txt = ch.get("text", "").strip()
                    if not txt:
                        continue
                    yield {"image": img, "text": txt, "bbox": ch.get("bbox", [0, 0, 0, 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_dir", default="runs/extract")
    ap.add_argument("--db_dir", default="runs/index")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    out = Path(args.db_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model)
    metas = []
    vecs = []
    for item in iter_texts(Path(args.extract_dir)):
        metas.append(item)
        vecs.append(model.encode(item["text"], normalize_embeddings=True))
    if not vecs:
        print("No chunks found.")
        return
    M = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatIP(M.shape[1])
    index.add(M)
    faiss.write_index(index, str(out / "chunks.faiss"))
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Indexed {len(metas)} chunks -> {out}")


if __name__ == "__main__":
    main()