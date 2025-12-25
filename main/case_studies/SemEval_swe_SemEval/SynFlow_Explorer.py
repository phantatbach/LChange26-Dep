import re
import os
from pathlib import Path
import sys

# Automate some steps:
# 1. Specify required input
# 2. Slot-path explorer


# Specify required input
# Add SynFlow to path in order to import modules
repo_root = "../"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Regex corpus_pattern to extract relevant information from CoNLL-U files
corpus_pattern = re.compile(
    r'([^\t]+)\t'      # word form
    r'([^\t]+)\t'      # lemma
    r'([^\t]+)\t'      # FULL POS or r'([^\t])[^\t]*\t' # POS-init (UPOS or XPOS)
    r'([^\t]+)\t'      # ID
    r'([^\t]+)\t'      # HEAD
    r'([^\t]+)'        # DEPREL
)

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

# Specify corpus and output folders
period = '1-2'
corpus_folder = f'../SemEval_swe_SemEval/merged_corpus/'
output_folder = Path(f'../case_studies/SemEval_swe_SemEval')

for selected_lemma in selected_lemmas:
    # Specify target lemma and part of speech
    target_lemma, selected_pos = selected_lemma.split('_')

    if selected_pos == 'N':
        target_pos = 'TAR'
    elif selected_pos == 'V':
        target_pos = 'TAR'
    elif selected_pos == 'A':
        target_pos = 'TAR'
    else:
        raise ValueError(f'Unknown part of speech: {selected_pos}')

    # Dont change below this line
    output_folder_lemma = output_folder / 'output' / f'{target_lemma}-{target_pos}-{period}'
    output_explorer = f'{output_folder_lemma}/Explorer'
    output_embedding = f'{output_folder_lemma}/Embedding'
    input_SCD = output_folder / 'input' / 'SCD' /f'{target_lemma}-{target_pos}-{period}'

    os.makedirs(output_explorer, exist_ok=True)
    os.makedirs(output_embedding, exist_ok=True)
    os.makedirs(input_SCD, exist_ok=True)

    from SynFlow.Explorer import spath_explorer
    
    dist = spath_explorer(
        corpus_folder=corpus_folder,
        target_lemma=target_lemma,
        target_pos=target_pos,
        max_length=1,
        top_n=50,
        pattern=corpus_pattern,
        output_folder=output_explorer
    )