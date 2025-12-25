#!/usr/bin/env python3
import os
from multiprocessing import Process
import pandas as pd
from tqdm import tqdm
import stanza

stanza.download('la', package='ittb', processors='tokenize,pos,lemma,depparse', verbose=False) # Latin dont have mvt
stanza.download('la', package='proiel', processors='tokenize,pos,lemma,depparse', verbose=False)

GPU_IDS = [
    0,
    2,
    ]

BASE = "./SemEval_lat"

# Jobs (corp_no, path_csv)
JOBS = [
    (1, f"{BASE}/corpus1/token/lat1.csv"),
    (2, f"{BASE}/corpus2/token/lat2.csv"),
]

# batch size
BATCH_SIZE = 100

# main
def worker(gpu_id: int, jobs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import stanza # Import for each GPU

    for corp_no, csv_path in jobs:
        if corp_no == 1:
            package = 'proiel'
        elif corp_no == 2:
            package = 'ittb'

        print(jobs, package)
        nlp = stanza.Pipeline(
                'la',
                package=package,
                processors='tokenize,pos,lemma,depparse', # Latin dont have mwt
                use_gpu=True,
                verbose=False,
                tokenize_no_ssplit=True
            )
        
        out_dir = f"{BASE}/corpus{corp_no}/reparsed"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/lat{corp_no}_reparsed.txt"

        token_df = pd.read_csv(csv_path)
        sents = token_df["sent"].tolist()
        reparsed_sents = []
        i = 0

        for start in tqdm(range(0, len(sents), BATCH_SIZE)):
            batch = sents[start:start + BATCH_SIZE]

            # Join batch
            texts = "\n\n".join(batch)

            doc = nlp(texts)
            for s in doc.sentences:
                lines = [f"<s id=lat{corp_no}_{i}>"]
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
