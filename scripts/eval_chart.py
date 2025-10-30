import argparse
import json
import re
from pathlib import Path


def load_lines(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def first_json_block(s: str):
    if not isinstance(s, str):
        return None
    m = re.search(r"\{.*\}", s, flags=re.S)
    return m.group(0) if m else None


def parse_pred_answer(pred_field):
    if pred_field is None:
        return None
    if isinstance(pred_field, dict):
        return str(pred_field.get("answer", "")).strip()
    if isinstance(pred_field, list):
        try:
            return str(pred_field[0]).strip()
        except Exception:
            return " ".join(map(str, pred_field)).strip()
    s = str(pred_field)
    jb = first_json_block(s)
    if jb:
        try:
            obj = json.loads(jb)
            if "answer" in obj:
                return str(obj["answer"]).strip()
        except Exception:
            pass
    return s.strip()


def unwrap_list_like(s: str) -> str:
    if not isinstance(s, str):
        return s
    m = re.match(r"^\s*\[\s*['\"](.+?)['\"]\s*\]\s*$", s.strip())
    return m.group(1) if m else s


def normalize_text(x: str):
    x = x.strip()
    x = x.replace(",", " ")
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" '"\t\r\n.")
    return x.lower()


def parse_number(x: str):
    if not x:
        return (None, False)
    is_percent = "%" in x
    s = x.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return (None, is_percent)
    return (float(m.group(0)), is_percent)


STOP = set("the a an of in on for to and or is are was were as at by with from".split())


def norm_tokens(x: str):
    x = x.lower()
    x = re.sub(r"[\[\]()%\",.:;]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    toks = [t.rstrip("s") for t in x.split() if t and t not in STOP]
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help="JSONL file containing chart predictions")
    ap.add_argument("--out", required=True, help="Output JSON file for metrics")
    ap.add_argument("--eps", type=float, default=0.02, help="Tolerance for numeric comparison")
    ap.add_argument("--lenient_text", type=int, default=1, help="Enable lenient text matching")
    args = ap.parse_args()
    total = 0
    n_eval = 0
    n_pred_nonnull = 0
    n_correct = 0
    n_num = 0
    n_num_correct = 0
    n_text = 0
    n_text_correct = 0
    bad_cases: list = []
    for ex in load_lines(Path(args.pred_jsonl)):
        gold_raw = str(ex.get("gold_answer") or ex.get("answer") or "").strip()
        gold = unwrap_list_like(gold_raw)
        q = str(ex.get("question", "")).strip()
        if not gold:
            continue
        total += 1
        pred = parse_pred_answer(ex.get("pred"))
        if pred is not None and pred != "":
            n_pred_nonnull += 1
        n_eval += 1
        g_num, g_pct = parse_number(gold)
        p_num, p_pct = parse_number(pred or "")
        if g_num is not None and p_num is not None:
            n_num += 1
            denom = max(abs(g_num), 1e-6)
            if denom > 1e-6:
                ok = abs(p_num - g_num) / denom <= args.eps
            else:
                ok = abs(p_num - g_num) <= args.eps
            n_num_correct += 1 if ok else 0
            if ok:
                n_correct += 1
            else:
                if len(bad_cases) < 10:
                    bad_cases.append({"image": ex.get("image"), "q": q, "gold": gold, "pred": pred})
        else:
            n_text += 1
            if args.lenient_text:
                gt = norm_tokens(gold)
                pt = norm_tokens(pred or "")
                jacc = len(set(gt) & set(pt)) / max(1, len(set(gt) | set(pt)))
                ok = (jacc >= 0.6) or (" ".join(pt) in " ".join(gt)) or (" ".join(gt) in " ".join(pt))
            else:
                ok = normalize_text(pred or "") == normalize_text(gold)
            n_text_correct += 1 if ok else 0
            if ok:
                n_correct += 1
            else:
                if len(bad_cases) < 10:
                    bad_cases.append({"image": ex.get("image"), "q": q, "gold": gold, "pred": pred})
    metrics = {
        "N_total_with_gold": total,
        "N_eval": n_eval,
        "Pred_nonnull_rate": round(n_pred_nonnull / max(total, 1), 4),
        "Acc_overall": round(n_correct / max(n_eval, 1), 4),
        "Acc_numeric": round(n_num_correct / max(n_num, 1), 4) if n_num > 0 else None,
        "Acc_text": round(n_text_correct / max(n_text, 1), 4) if n_text > 0 else None,
        "N_numeric": n_num,
        "N_text": n_text,
        "eps": args.eps,
        "lenient_text": args.lenient_text,
        "examples_bad": bad_cases,
    }
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()