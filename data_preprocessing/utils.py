# Helper functions

import re
# Search in txt
def open_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def search_in_txt(content, target):
    """
    Sử dụng biểu thức chính quy (re.findall) để tìm và đếm số lần xuất hiện.
    Target là mẫu regex.
    """
    # Tìm tất cả các lần xuất hiện của mẫu regex trong nội dung.
    # Sử dụng re.MULTILINE để đảm bảo $ khớp với cuối dòng
    matches = re.findall(target, content, flags=re.MULTILINE)
    return len(matches)

def replace_in_txt(content, target, replacement):
    """
    Sử dụng biểu thức chính quy (re.sub) để thay thế.
    Target và replacement phải là các chuỗi regex hợp lệ.
    """
    # re.sub(pattern, repl, string, count=0, flags=0)
    # flags=re.MULTILINE để đảm bảo $ (cuối dòng) hoạt động chính xác
    new_content = re.sub(target, replacement, content, flags=re.MULTILINE)
    return new_content

def save_to_txt(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def return_stanza_parsed_tags(reparsed_sent, selected_lemma):
    # selected_lemma kiểu "attack_v" -> lấy "attack"
    target = selected_lemma.rsplit('_', 1)[0]

    results = []

    for line in reparsed_sent.split('\n'):
        if not line or line.startswith('<s ') or line.startswith('</s>'):
            continue

        cols = line.split('\t')
        if len(cols) < 3:
            continue

        _, lemma, upos = cols[0], cols[1], cols[2]

        if lemma == target:
            # giữ format cũ
            results.append(f"{lemma}\t{upos}")

    return results