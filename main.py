import pdfplumber
import spacy
from BTrees.OOBTree import OOBTree

nlp = spacy.load("en_core_web_sm")

class Colors:
    HEADER = '\033[95m'  # Purple for headers
    BLUE = '\033[94m'  # Blue
    CYAN = '\033[96m'  # Cyan
    GREEN = '\033[92m'  # Green
    YELLOW = '\033[93m'  # Yellow
    RED = '\033[91m'  # Red
    RESET = '\033[0m'  # Reset to normal
    BOLD = '\033[1m'  # Bold
    UNDERLINE = '\033[4m'  # Underline


def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + ' '
    return text


def tokenize(text):
    return nlp(text)


def lemmatize(doc):
    lemma_terms = [
        token.text
        for token in doc
        if token.is_alpha
    ]
    return lemma_terms


def build_document_btree(pdf_files, lemma_sets):
    """Build B-Tree to store documents with their metadata"""
    doc_btree = OOBTree()

    for doc_id, (pdf_path, lemma_list) in enumerate(zip(pdf_files, lemma_sets)):
        doc_btree[doc_id] = {
            'filename': pdf_path,
            'terms': lemma_list,
            'term_count': len(set(lemma_list))
        }

    return doc_btree


def build_inverted_index(lemma_sets):
    """Build standard inverted index with term, frequency, and postings"""
    inverted_index = OOBTree()

    for doc_id, lemmas in enumerate(lemma_sets):
        for lemma in lemmas:
            key = lemma
            if key not in inverted_index:
                inverted_index[key] = {
                    'frequency': 0,
                    'postings': set()
                }
            inverted_index[key]['frequency'] += 1
            inverted_index[key]['postings'].add(doc_id)

    return inverted_index


def build_btree_structure(keys, order=4):
    """Build a proper B-Tree structure from keys"""
    if len(keys) == 0:
        return None

    sorted_keys = sorted(keys)

    if len(sorted_keys) <= order:
        return {
            'keys': sorted_keys,
            'children': [],
            'level': 0
        }


    children = []
    for i in range(0, len(sorted_keys), order):
        chunk = sorted_keys[i:i + order]
        children.append({
            'keys': chunk,
            'children': [],
            'level': 0
        })

    current_level = children
    level = 1

    while len(current_level) > order:
        new_level = []
        for i in range(0, len(current_level), order):
            chunk = current_level[i:i + order]
            parent_keys = [child['keys'][0] for child in chunk]
            new_level.append({
                'keys': parent_keys,
                'children': chunk,
                'level': level
            })
        current_level = new_level
        level += 1

    if len(current_level) == 1:
        root = current_level[0]
        root['level'] = level
        return root
    else:
        return {
            'keys': [node['keys'][0] for node in current_level],
            'children': current_level,
            'level': level
        }


def display_btree_layered(btree_struct, title="B-Tree Structure"):
    """Display B-Tree in console with proper layers"""

    if not btree_struct:
        print(f"{Colors.RED}B-Tree is empty!{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 100}{Colors.RESET}\n")

    levels_dict = {}

    def collect_by_level(node):
        level = node['level']
        if level not in levels_dict:
            levels_dict[level] = []
        levels_dict[level].append(node)

        for child in node['children']:
            collect_by_level(child)

    collect_by_level(btree_struct)

    max_level = max(levels_dict.keys())

    for level in range(max_level, -1, -1):
        nodes_at_level = levels_dict[level]

        if level == max_level:
            level_name = f"{Colors.BOLD}{Colors.YELLOW}ROOT LEVEL{Colors.RESET}"
        elif level == 0:
            level_name = f"{Colors.BOLD}{Colors.GREEN}LEAF LEVEL{Colors.RESET}"
        else:
            level_name = f"{Colors.BOLD}{Colors.BLUE}INTERNAL LEVEL {level}{Colors.RESET}"

        print(f"{level_name} (Total Nodes: {len(nodes_at_level)})")
        print(f"{Colors.CYAN}{'─' * 100}{Colors.RESET}")


        for node_idx, node in enumerate(nodes_at_level):
            keys_display = ' │ '.join([
                f"{Colors.YELLOW}{str(k)[:15]}{Colors.RESET}"
                for k in node['keys']
            ])

            num_children = len(node['children'])
            children_info = f" → {num_children} children" if num_children > 0 else " (Leaf)"

            print(
                f"  {Colors.BOLD}Node {node_idx}:{Colors.RESET} [{keys_display}{Colors.CYAN}]{Colors.RESET}{children_info}")

        print()


def display_btree_with_structure(inverted_index, order=4):
    """Display B-Tree structure based on inverted index terms"""

    if len(inverted_index) == 0:
        print(f"{Colors.RED}Inverted Index is empty!{Colors.RESET}")
        return

    terms = sorted(list(inverted_index.keys()))
    btree_struct = build_btree_structure(terms, order=order)

    display_btree_layered(btree_struct, f"B-Tree for Inverted Index Terms (Order: {order}, Total Terms: {len(terms)})")

pdf_files = ['./ow07.pdf', './networking.pdf', './ow04.pdf']
lemma_sets = []

print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 100}{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}Processing PDF Documents{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 100}{Colors.RESET}\n")

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"{Colors.CYAN}Processing Document {idx}: {pdf_path}{Colors.RESET}")
    print(f"{Colors.YELLOW}Tokenizing ...{Colors.RESET}")

    text = extract_text_from_pdf(pdf_path)
    doc = tokenize(text)

    print(f"{Colors.GREEN}List of words:{Colors.RESET}")
    words_list = [token.text for token in doc if token.is_alpha]
    print(f"  {words_list[:100]}...")

    lemma_terms = lemmatize(doc)
    lemma_sets.append(lemma_terms)

    print(f"{Colors.BLUE}List of lemmatized words (terms):{Colors.RESET}")
    lemma_list = sorted(list(set(lemma_terms)), key=str.lower)
    print(f"  {lemma_list[:100]}...")
    print()

print(f"{Colors.GREEN}{'=' * 100}{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.YELLOW}Creating B-Tree for documents start{Colors.RESET}")
doc_btree = build_document_btree(pdf_files, lemma_sets)
print(f"{Colors.BOLD}{Colors.CYAN}Document B-Tree created successfully!{Colors.RESET}")
print(f"{Colors.CYAN}Total documents stored: {len(doc_btree)}{Colors.RESET}\n")

print(f"{Colors.BOLD}{Colors.YELLOW}Creating inverted index start{Colors.RESET}")
inverted_index = build_inverted_index(lemma_sets)
print(f"{Colors.BOLD}{Colors.CYAN}Inverted index created successfully!{Colors.RESET}")
print(f"{Colors.CYAN}Total unique terms: {len(inverted_index)}{Colors.RESET}\n")


display_btree_with_structure(inverted_index, order=4)
print(f"{Colors.BLUE}Inverted Index (Term | Frequency | Postings):{Colors.RESET}")
print(f"{Colors.YELLOW}{'Term':<30} | {'Frequency':<12} | {'Postings':<20}{Colors.RESET}")
print(f"{Colors.YELLOW}{'-' * 30}-+-{'-' * 12}-+-{'-' * 20}{Colors.RESET}")

all_terms = sorted(list(inverted_index.keys()))
for term in all_terms:
    frequency = inverted_index[term]['frequency']
    postings = sorted(list(inverted_index[term]['postings']))
    print(f"{Colors.CYAN}{term:<30}{Colors.RESET} | {frequency:<12} | {str(postings):<20}")
print()

print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 100}{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}Search Section{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 100}{Colors.RESET}\n")

isContinue = True
while isContinue:
    query = input(f"{Colors.BOLD}{Colors.YELLOW}Enter a term to search (type 'exit' to quit): {Colors.RESET}")

    if query.lower() == 'exit':
        isContinue = False
        print(f"{Colors.BOLD}{Colors.GREEN}Goodbye!{Colors.RESET}")
    else:
        result = inverted_index.get(query, {})
        print(f"{Colors.BOLD}{Colors.CYAN}Search Results for \"{query}\":{Colors.RESET}")

        if result:
            frequency = result['frequency']
            postings = sorted(list(result['postings']))
            print(f"{Colors.GREEN}  Frequency: {frequency}{Colors.RESET}")
            print(f"{Colors.CYAN}  Postings (Document IDs): {postings}{Colors.RESET}")
            print(f"{Colors.CYAN}  Found in {len(postings)} document(s){Colors.RESET}")
        else:
            print(f"{Colors.RED}  No documents found containing this term.{Colors.RESET}")
        print()