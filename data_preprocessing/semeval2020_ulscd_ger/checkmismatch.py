import pandas as pd
import os
import stanza
import re
import sys
from tqdm import tqdm

sys.path.append('../data_preprocessing')
from utils import open_txt, save_to_txt, search_in_txt, replace_in_txt, return_stanza_parsed_tags

# Mismatch report
def convert_org_stanza(org_lemma_pos):
    # Chuyển từ dạng gốc sang dạng stanza
    org_lemma = org_lemma_pos.rsplit('_')[0]
    org_pos = org_lemma_pos.split('_')[-1]

    stanza_lemma = org_lemma
    stanza_pos = None
    if org_pos == 'N':
        stanza_pos = 'NOUN'
    elif org_pos == 'V':
        stanza_pos = 'VERB'
    elif org_pos == 'A':
        stanza_pos = 'ADJ'
    
    return stanza_lemma, stanza_pos

selected_lemmas = [
    'abbauen_V',
    'abdecken_V',
    'abgebrüht_A',
    'Abgesang_N',
    'Ackergerät_N',
    'Armenhaus_N',
    'artikulieren_V',
    'aufrechterhalten_V',
    'Ausnahmegesetz_N',
    'ausspannen_V',
    'beimischen_V',
    'Dynamik_N',
    'Einreichung_N',
    'Eintagsfliege_N',
    'Engpaß_N',
    'Entscheidung_N',
    'Festspiel_N',
    'Frechheit_N',
    'Fuß_N',
    'Gesichtsausdruck_N',
    'Knotenpunkt_N',
    'Kubikmeter_N',
    'Lyzeum_N',
    'Manschette_N',
    'Mißklang_N',
    'Mulatte_N',
    'Naturschönheit_N',
    'Ohrwurm_N',
    'Pachtzins_N',
    'packen_V',
    'Rezeption_N',
    'Schmiere_N',
    'Seminar_N',
    'Sensation_N',
    'Spielball_N',
    'Tier_N',
    'Titel_N',
    'Tragfähigkeit_N',
    'Truppenteil_N',
    'überspannen_V',
    'Unentschlossenheit_N',
    'verbauen_V',
    'vergönnen_V',
    'voranstellen_V',
    'vorliegen_V',
    'vorweisen_V',
    'weitgreifend_A',
    'zersetzen_V'
    ]

# MAIN FUNCTION TO CHECK FOR MISMACHES

# Create 1 dictionary to store mismatched for all selected_lemmas
report_mismatches = {}

for corp_no in [1, 2]:
    report_mismatches[corp_no] = {}

    # Org lemma file
    org_parsed_file = f'./SemEval_ger/corpus{corp_no}/lemma/ger{corp_no}.csv'
    org_df = pd.read_csv(org_parsed_file)

    # Reparsed file
    reparsed_file = f'./SemEval_ger/corpus{corp_no}/reparsed/ger{corp_no}_reparsed.txt'
    with open(reparsed_file, 'r') as f:
        reparsed_content = f.read()
        # Separate sentences
        reparsed_sents = reparsed_content.strip().split("\n\n")
        reparsed_sent_count = len(reparsed_sents)

    # Check no of sent
    org_df['sent'] = org_df['sent'].fillna('') # Fillna because there are some empty lines
    org_sents = org_df['sent'].tolist() 
    org_sent_count = len(org_sents)

    if org_sent_count != reparsed_sent_count:
        report_mismatches[corp_no]['Mismatched numbers of sentences'] = f"org={org_sent_count}, reparsed={reparsed_sent_count}"
        continue
    
    # Statistics for each lemma
    for selected_lemma in tqdm(selected_lemmas):
        selected_lemma_base = selected_lemma.rsplit('_', 1)[0] # In other datasets than English, there is no _POS in the target

        # Whole file statistics:
        mismatch_sent = 0
        org_miss_lemma_count = 0
        reparsed_miss_lemma_count = 0
        report_mismatches[corp_no][selected_lemma] = []
        
        # Count selected_lemma_base in file
        pattern = rf'\b{re.escape(selected_lemma_base)}\b'

        org_lemma_count = org_df['sent'].str.count(pattern).sum()
        if org_lemma_count == 0: # Avoid division by zero error if the lemma is not found
            org_lemma_count = 1
        
        # Individual sentence check
        for i in range(org_sent_count):
            org_sent = org_sents[i]
            reparsed_sent = reparsed_sents[i]

            # Count selected_lemma_base occurrences in original sentence
            org_count = len(re.findall(pattern, org_sent))

            # Count selected_lemma occurrences in reparsed sentence
            stanza_lemma, stanza_pos = convert_org_stanza(selected_lemma)
            stanza_format = f'\t{stanza_lemma}\t{stanza_pos}\t'
            reparsed_count = reparsed_sent.count(stanza_format)

            # Return stanza tags for the selected_lemma
            stanza_tags = return_stanza_parsed_tags(reparsed_sent, selected_lemma)

            if org_count != reparsed_count:
                mismatch_sent += 1
                (report_mismatches[corp_no][selected_lemma].append(
                    f"Mismatch sentence:{i}, "
                    f"org={org_count}, "
                    f"reparsed={reparsed_count}, "
                    f"org_sent='{org_sent}, "
                    f"stanza_pos={stanza_tags}"))
                
                if org_count >= reparsed_count:
                    reparsed_miss_lemma_count += (org_count - reparsed_count)
                elif org_count < reparsed_count:
                    org_miss_lemma_count += (reparsed_count - org_count)

        # Whole file statistics:
        if mismatch_sent != 0:
            report_mismatches[corp_no][selected_lemma].append(f"Total mismatched sentences: {mismatch_sent} ({mismatch_sent/org_sent_count*100:.2f}%)")
            report_mismatches[corp_no][selected_lemma].append(f"Total missing lemma in original file (compared to org): {org_miss_lemma_count} ({org_miss_lemma_count/ (org_lemma_count)*100:.2f}%)")
            report_mismatches[corp_no][selected_lemma].append(f"Total missing lemma in reparsed file (compared to org): {reparsed_miss_lemma_count} ({reparsed_miss_lemma_count/org_lemma_count*100:.2f}%)")

with open('./SemEval_ger/mismatch_report.txt', 'w') as f:
    # Write the report_mismatches dictionary to the file beautifully
    for corp_no in report_mismatches:
        f.write(f"Corpus {corp_no}:\n")
        for selected_lemma in report_mismatches[corp_no]:
            f.write(f"Lemma: {selected_lemma}\n")
            for mismatch in report_mismatches[corp_no][selected_lemma]:
                f.write(f"{mismatch}\n")
            f.write("\n")