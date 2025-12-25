import re
import os
import pandas as pd
from pathlib import Path
import sys
import json

selected_lemmas = [
    'aktiv_A',
    'annandag_N',
    'antyda_V',
    'bearbeta_V',
    'bedömande_N',
    'beredning_N',
    'blockera_V',
    'bolagsstämma_N',
    'bröllop_N',
    'by_N',
    'central_A',
    'färg_N',
    'förhandling_N',
    'gagn_N',
    'granskare_N',
    'kemisk_A',
    'kokärt_N',
    'konduktör_N',
    'krita_N',
    'ledning_N',
    'medium_N',
    'motiv_N',
    'notis_N',
    'studie_N',
    'undertrycka_V',
    'uppfattning_N',
    'uppfostran_N',
    'uppläggning_N',
    'uträtta_V',
    'vaktmästare_N',
    'vegetation_N'
]

# Automate all the steps for all selected lemmas

# Specify required input
# Add SynFlow to path in order to import modules
repo_root = "../"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

results = {}

# Specify corpus and output folders
period = '1-2'
corpus_folder = f'../SemEval_swe/merged_corpus/'
output_folder = Path(f'../case_studies/SemEval_swe')

for selected_lemma in selected_lemmas:
    # Specify target lemma and part of speech
    target_lemma, selected_pos = selected_lemma.split('_')

    if selected_pos == 'N':
        target_pos = 'NOUN'
    elif selected_pos == 'V':
        target_pos = 'VERB'
    elif selected_pos == 'A':
        target_pos = 'ADJ'
    else:
        raise ValueError(f'Unknown part of speech: {selected_pos}')

    # Input target
    keyword_string = f'{target_lemma}\t{target_pos}' # Or you can use the full POS for precision (e.g., {target_lemma}\tNOUN)

    # Pattern of the file names and regex patterns of the CONLLU file
    fname_pattern = re.compile(
        r'^swe[12]_reparsed\.txt$'
    )
    corpus_pattern = re.compile(
        r'([^\t]+)\t'      # word form
        r'([^\t]+)\t'      # lemma
        r'([^\t]+)\t'      # FULL POS or r'([^\t])[^\t]*\t' # POS-init (UPOS or XPOS)
        r'([^\t]+)\t'      # ID
        r'([^\t]+)\t'      # HEAD
        r'([^\t]+)'        # DEPREL
    )

    # The path to the slot count JSON file
    slot_json_path = f'../case_studies/SemEval_swe/output/{target_lemma}-{target_pos}-1-2/Explorer/{target_lemma}_{target_pos}_spaths.json'

    # Dont change below this line
    output_folder_lemma = output_folder / 'output' / f'{target_lemma}-{target_pos}-{period}'
    output_explorer = f'{output_folder_lemma}/Explorer'
    output_embedding = f'{output_folder_lemma}/Embedding'
    input_SCD = output_folder / 'input' / 'SCD' /f'{target_lemma}-{target_pos}-{period}'

    os.makedirs(output_explorer, exist_ok=True)
    os.makedirs(output_embedding, exist_ok=True)
    os.makedirs(input_SCD, exist_ok=True)

    # Slot Frequencies
    # Token counts for normalisation
    from SynFlow.SCD import count_keyword_tokens_by_period
    token_counts = count_keyword_tokens_by_period(corpus_folder, keyword_string,
                                                fname_pattern=fname_pattern)
    print(token_counts)

    from SynFlow.SCD import plot_freq_top_union_slots_by_period

    plot_freq_top_union_slots_by_period(
        json_path=slot_json_path,
        top_n=10,
        normalized=False,
        relative=False,
        token_counts=token_counts,
    )

    from SynFlow.SCD import freq_all_slots_by_period
    slot_raw_freq_df = freq_all_slots_by_period(json_path=slot_json_path).T
    slot_raw_freq_df.head(30)

    # Get all slots in the corect format
    from SynFlow.Explorer.sfiller_df import get_all_slots
    all_slots = get_all_slots(slot_raw_freq_df)

    # Building a slot filler df
    from SynFlow.Explorer import build_sfiller_df

    df_slots = build_sfiller_df(
        corpus_folder=corpus_folder,
        template=all_slots, 
        target_lemma=target_lemma,
        target_pos=target_pos,
        pattern=corpus_pattern,
        # freq_path='..//RSC/lemma_pos_init_freq.txt', # Be sure that the freq_path matches that of the filter format
        # freq_min=1,
        # freq_max=100_000_000,
        filtered_pos=[],
        filler_format='lemma/pos', # lemma/deprel or 'lemma/pos'
        output_folder= output_explorer
        )
    
    all_sfillers_csv_path = f'../case_studies/SemEval_swe/output/{target_lemma}-{target_pos}-1-2/Explorer/{target_lemma}_samples_sfillerdf_all.csv'

    # Calculate divergences of all slots
    from SynFlow.SCD import consecutive_JSD_dict
    sfiller_df_path = all_sfillers_csv_path

    consecutive_JSD_dictionary = consecutive_JSD_dict(all_sfillers_csv_path=sfiller_df_path,
                     min_freq=1,
                     mode='all' # or 'data_only' if you want to skip the empty periods
                     )
    
    # Store results
    results[selected_lemma] = consecutive_JSD_dictionary

# Transform results
# Remove slots with empty or 0 JSD scores

def transform_results(results):
    transformed_results = {}

    for lemma, slot_per_jsd in results.items():
        transformed_results[lemma] = {}
        for slot, per_jsd in slot_per_jsd.items():
            jsd = next(iter(per_jsd.values()), None)
            if jsd is None:
                continue
            elif jsd > 0:
                transformed_results[lemma][slot] = jsd

    return transformed_results

transformed_results = transform_results(results)
transformed_results

# Dump results
with open("./result_semeval.json", "w", encoding="utf-8") as f:
    json.dump(transformed_results, f, ensure_ascii=False, indent=2, sort_keys=True)