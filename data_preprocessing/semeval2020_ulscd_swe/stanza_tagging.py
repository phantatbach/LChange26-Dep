#!/usr/bin/env python3
import os
from multiprocessing import Process
import pandas as pd
from tqdm import tqdm
import stanza

stanza.download('sv', processors='tokenize,pos,lemma,depparse', verbose=False) # Download stanza once. Swedish dont have mwt

GPU_IDS = [
    0,
    1,
    ]

BASE = "./SemEval_swe"

# Job (corp_no, path_csv)
JOBS = [
    (1, f"{BASE}/corpus1/token/swe1.csv"),
    (2, f"{BASE}/corpus2/token/swe2.csv"),
]

# Batch size
BATCH_SIZE = 100

# Main
def worker(gpu_id: int, jobs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import stanza # Import for each GPU
    nlp = stanza.Pipeline(
        'sv',
        processors='tokenize,pos,lemma,depparse', # Swedish dont have mwt
        use_gpu=True,
        verbose=False,
        tokenize_no_ssplit=True
    )

    for corp_no, csv_path in jobs:
        out_dir = f"{BASE}/corpus{corp_no}/reparsed"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/swe{corp_no}_reparsed.txt"

        token_df = pd.read_csv(csv_path)
        sents = token_df["sent"].tolist()
        reparsed_sents = []
        i = 0

        for start in tqdm(range(0, len(sents), BATCH_SIZE)):
            batch = sents[start:start + BATCH_SIZE]

            # Join sentences
            texts = "\n\n".join(batch)

            doc = nlp(texts)
            for s in doc.sentences:
                lines = [f"<s id=swe{corp_no}_{i}>"]
                for w in s.words:
                    lines.append(
                        f"{w.text}\t{w.lemma}\t{w.upos}\t{w.id}\t{w.head}\t{w.deprel}"
                    )
                lines.append("</s>")
                reparsed_sents.append("\n".join(lines))
                i += 1

        with open(out_path, "w") as f:
            f.write("\n\n".join(reparsed_sents))

def split_round_robin(items, k):
    buckets = [[] for _ in range(k)]
    for idx, it in enumerate(items):
        buckets[idx % k].append(it)
    return buckets

if __name__ == "__main__":
    num_workers = min(len(GPU_IDS), len(JOBS))
    buckets = split_round_robin(JOBS, num_workers)

    procs = []
    for w_idx in range(num_workers):
        p = Process(target=worker, args=(GPU_IDS[w_idx], buckets[w_idx]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
