import math
import re
import pdfplumber
import spacy
from nltk.stem import PorterStemmer
from BTrees.OOBTree import OOBTree
from collections import defaultdict, Counter

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()


class Colors:
    BLUE = '\033[94m'  # Blue
    CYAN = '\033[96m'  # Cyan
    GREEN = '\033[92m'  # Green
    YELLOW = '\033[93m'  # Yellow
    RED = '\033[91m'  # Red
    RESET = '\033[0m'  # Reset to normal


def colorizePrint(color_name, message):
    color = getattr(Colors, color_name, Colors.RESET)
    print(f"{color}{message}{Colors.RESET}")


def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n\n'  # separate pages by blank line (helps paragraph splitting)
    return text


def tokenize(text):
    return nlp(text)


def stem_tokens(doc):
    """Return list of Porter stems (lowercased) for alpha tokens in a spaCy doc."""
    return [
        stemmer.stem(token.text.lower())
        for token in doc
        if token.is_alpha
    ]


def build_document_btree(pdf_files, stem_sets):
    """Build B-Tree to store documents with their metadata"""
    doc_btree = OOBTree()

    for doc_id, (pdf_path, stem_list) in enumerate(zip(pdf_files, stem_sets)):
        doc_btree[doc_id] = {
            'filename': pdf_path,
            'terms': stem_list,
            'term_count': len(set(stem_list))
        }

    return doc_btree


def build_inverted_index(stem_sets):
    """Build standard inverted index with term, frequency, and postings"""
    inverted_index = OOBTree()

    for doc_id, stems in enumerate(stem_sets):
        for stem in stems:
            key = stem.lower()
            if key not in inverted_index:
                inverted_index[key] = {
                    'frequency': 0,
                    'postings': set()
                }
            inverted_index[key]['frequency'] += 1
            inverted_index[key]['postings'].add(doc_id)

    return inverted_index


def build_btree_structure(keys, order=4):
    """Build a simple B-tree-like structure for display"""
    if len(keys) == 0:
        return None
    sorted_keys = sorted(keys)
    if len(sorted_keys) <= order:
        return {'keys': sorted_keys, 'children': [], 'level': 0}
    children = []
    for i in range(0, len(sorted_keys), order):
        chunk = sorted_keys[i:i + order]
        children.append({'keys': chunk, 'children': [], 'level': 0})
    current_level = children
    level = 1
    while len(current_level) > order:
        new_level = []
        for i in range(0, len(current_level), order):
            chunk = current_level[i:i + order]
            parent_keys = [child['keys'][0] for child in chunk]
            new_level.append({'keys': parent_keys, 'children': chunk, 'level': level})
        current_level = new_level
        level += 1
    if len(current_level) == 1:
        root = current_level[0]
        root['level'] = level
        return root
    return {'keys': [node['keys'][0] for node in current_level], 'children': current_level, 'level': level}


def display_btree_layered(btree_struct, title="B-Tree Structure"):
    if not btree_struct:
        colorizePrint("RED", "B-Tree is empty!")
        return
    colorizePrint("CYAN", f"\n{title}")
    colorizePrint("CYAN", "=" * 80 + "\n")
    levels = {}

    def collect(node):
        lvl = node['level']
        levels.setdefault(lvl, []).append(node)
        for c in node.get('children', []):
            collect(c)

    collect(btree_struct)
    max_level = max(levels.keys())
    for level in range(max_level, -1, -1):
        nodes = levels[level]
        if level == max_level:
            colorizePrint("YELLOW", f"ROOT LEVEL (Total Nodes: {len(nodes)})")
        elif level == 0:
            colorizePrint("GREEN", f"LEAF LEVEL (Total Nodes: {len(nodes)})")
        else:
            colorizePrint("BLUE", f"INTERNAL LEVEL {level} (Total Nodes: {len(nodes)})")
        colorizePrint("CYAN", "─" * 80)
        for i, node in enumerate(nodes):
            keys_display = ' │ '.join([str(k)[:15] for k in node['keys']])
            children_info = f" → {len(node.get('children', []))} children" if node.get('children') else " (Leaf)"
            colorizePrint("BOLD", f"  Node {i}:")
            colorizePrint("YELLOW", f"    [{keys_display}] {children_info}")
        print()


def display_btree_with_structure(inverted_index, order=4):
    if len(inverted_index) == 0:
        colorizePrint("RED", "Inverted Index is empty!")
        return
    terms = sorted(list(inverted_index.keys()))
    btree_struct = build_btree_structure(terms, order=order)
    display_btree_layered(btree_struct, f"B-Tree for Inverted Index Terms (Order: {order}, Total Terms: {len(terms)})")


# ---- Main processing ----
pdf_files = ['./ow07.pdf', './networking.pdf', './ow04.pdf', './doc3.pdf']  # update paths as needed
stem_sets = []
raw_texts = []  # store full raw text per document (for phrase/sentence/paragraph checks)

colorizePrint("HEADER", "=" * 80)
colorizePrint("HEADER", "Processing PDF Documents")
colorizePrint("HEADER", "=" * 80 + "\n")

for idx, pdf_path in enumerate(pdf_files, 1):
    colorizePrint("CYAN", f"Processing Document {idx}: {pdf_path}")
    colorizePrint("YELLOW", "Tokenizing ...")
    text = extract_text_from_pdf(pdf_path)
    raw_texts.append(text)
    doc = tokenize(text)
    colorizePrint("GREEN", "List of words:")
    words_list = [token.text for token in doc if token.is_alpha]
    print(" ", words_list[:100], "...")
    stem_terms = stem_tokens(doc)
    stem_sets.append(stem_terms)
    colorizePrint("BLUE", "List of stemmed words (terms):")
    stem_list = sorted(list(set(stem_terms)), key=str.lower)
    print("  ", stem_list[:100], "...")
    print()

colorizePrint("GREEN", "=" * 80)
colorizePrint("YELLOW", "Creating B-Tree for documents start")
doc_btree = build_document_btree(pdf_files, stem_sets)
colorizePrint("CYAN", "Document B-Tree created successfully!")
colorizePrint("CYAN", f"Total documents stored: {len(doc_btree)}\n")

colorizePrint("YELLOW", "Creating inverted index start")
inverted_index = build_inverted_index(stem_sets)
colorizePrint("CYAN", "Inverted index created successfully!")
colorizePrint("CYAN", f"Total unique terms: {len(inverted_index)}\n")

display_btree_with_structure(inverted_index, order=4)

colorizePrint("BLUE", "Inverted Index (Term | Frequency | Postings):")
colorizePrint("YELLOW", f"{'Term':<30} | {'Frequency':<12} | {'Postings':<20}")
colorizePrint("YELLOW", f"{'-' * 30}-+-{'-' * 12}-+-{'-' * 20}")

all_terms = sorted(list(inverted_index.keys()))
for term in all_terms:
    frequency = inverted_index[term]['frequency']
    postings = sorted(list(inverted_index[term]['postings']))
    colorizePrint("CYAN", f"{term:<30} | {frequency:<12} | {str(postings):<20}")
print()

colorizePrint("HEADER", "=" * 80)
colorizePrint("HEADER", "Search Section")
colorizePrint("HEADER", "=" * 80 + "\n")


# -------------------------
# Precompute structures for retrieval models
# -------------------------
N_DOCS = len(pdf_files)
# term -> {doc_id: tf}
term_doc_tf = defaultdict(lambda: defaultdict(int))
doc_lengths = [0] * N_DOCS
for doc_id, stems in enumerate(stem_sets):
    lowered = [t.lower() for t in stems]
    doc_lengths[doc_id] = len(lowered)
    freqs = Counter(lowered)
    for term, tf in freqs.items():
        term_doc_tf[term][doc_id] = tf

term_df = {term: len(doc_map) for term, doc_map in term_doc_tf.items()}

# collection frequency and total tokens (for probabilistic / language model)
collection_term_freq = {term: sum(doc_map.values()) for term, doc_map in term_doc_tf.items()}
total_collection_terms = sum(doc_lengths) if doc_lengths else 0

# TF-IDF weights per term per doc (precompute for vector space)
doc_term_tfidf = defaultdict(dict)
doc_vector_norm = [0.0] * N_DOCS
for term, doc_map in term_doc_tf.items():
    df = term_df.get(term, 0)
    if df == 0:
        continue
    idf = math.log(N_DOCS / df) if df > 0 else 0.0
    for doc_id, tf in doc_map.items():
        tf_w = 1.0 + math.log(tf)
        w = tf_w * idf
        doc_term_tfidf[term][doc_id] = w
        doc_vector_norm[doc_id] += w * w
for i in range(N_DOCS):
    doc_vector_norm[i] = math.sqrt(doc_vector_norm[i]) if doc_vector_norm[i] > 0 else 0.0


# -------------------------
# Helper: intersection of posting sets
# -------------------------
def intersection(list_of_sets):
    """Return intersection of list_of_sets (iterable of sets)."""
    if not list_of_sets:
        return set()
    res = None
    for s in list_of_sets:
        if res is None:
            res = set(s)
        else:
            res &= set(s)
        if not res:
            break
    return res if res is not None else set()


# -------------------------
# STANDARD BOOLEAN (AND/OR/NOT, parentheses)
# -------------------------
def _tokenize_boolean_query(raw_query: str):
    q = raw_query.replace('(', ' ( ').replace(')', ' ) ')
    parts = q.split()
    tokens = []
    for p in parts:
        up = p.upper()
        if up in ("AND", "OR", "NOT", "(", ")"):
            tokens.append(up)
        else:
            # normalize term to lowercase, keep alphas, then stem
            doc = nlp(p)
            term_tokens = [stemmer.stem(t.text.lower()) for t in doc if t.is_alpha]
            if term_tokens:
                tokens.append(term_tokens[0])
    return tokens


def _infix_to_postfix(tokens):
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    output = []
    stack = []
    for tok in tokens:
        if tok == '(':
            stack.append(tok)
        elif tok == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()
        elif tok in ("AND", "OR", "NOT"):
            while stack and stack[-1] != '(' and prec.get(stack[-1], 0) >= prec.get(tok, 0):
                output.append(stack.pop())
            stack.append(tok)
        else:
            output.append(tok)
    while stack:
        output.append(stack.pop())
    return output


def _eval_postfix(postfix_tokens):
    stack = []
    for tok in postfix_tokens:
        if tok == "NOT":
            operand = stack.pop() if stack else set()
            stack.append(set(range(N_DOCS)).difference(operand))
        elif tok in ("AND", "OR"):
            if len(stack) < 2:
                return set()
            b = stack.pop()
            a = stack.pop()
            if tok == "AND":
                stack.append(a.intersection(b))
            else:
                stack.append(a.union(b))
        else:
            postings = set(term_doc_tf.get(tok, {}).keys())
            stack.append(postings)
    return stack.pop() if stack else set()


def standard_boolean_model(query):
    colorizePrint("CYAN", f"[STANDARD BOOLEAN] Query: {query}")
    tokens = _tokenize_boolean_query(query)
    if not tokens:
        colorizePrint("RED", "No valid tokens in query.")
        return []
    postfix = _infix_to_postfix(tokens)
    result = _eval_postfix(postfix)
    if not result:
        colorizePrint("RED", "No documents matched.")
        return []
    ranked = sorted(result)
    colorizePrint("GREEN", f"Matched documents: {ranked}")
    return ranked


# -------------------------
# EXTENDED BOOLEAN (Westlaw-like subset)
# -------------------------
def _doc_terms_list(doc_id):
    return [t.lower() for t in stem_sets[doc_id]]


def _doc_text(doc_id):
    return raw_texts[doc_id]


def _find_phrase_in_doc(phrase, doc_id):
    terms = [stemmer.stem(t.text.lower()) for t in tokenize(phrase) if t.is_alpha]
    if not terms:
        return False
    doc_terms = _doc_terms_list(doc_id)
    n = len(terms)
    for i in range(len(doc_terms) - n + 1):
        if doc_terms[i:i + n] == terms:
            return True
    return False


def _within_k_words(term1, term2, k, doc_id):
    doc_terms = _doc_terms_list(doc_id)
    positions1 = [i for i, t in enumerate(doc_terms) if t == term1]
    positions2 = [i for i, t in enumerate(doc_terms) if t == term2]
    for p1 in positions1:
        for p2 in positions2:
            if abs(p1 - p2) - 1 <= k:
                return True
    return False


def _same_sentence(term1, term2, doc_id):
    text = _doc_text(doc_id)
    doc_spacy = tokenize(text)
    for sent in doc_spacy.sents:
        sent_terms = [stemmer.stem(t.text.lower()) for t in sent if t.is_alpha]
        if term1 in sent_terms and term2 in sent_terms:
            return True
    return False


def _same_paragraph(term1, term2, doc_id):
    paras = [p.strip() for p in _doc_text(doc_id).split('\n\n') if p.strip()]
    for p in paras:
        pdoc = tokenize(p)
        pterms = [stemmer.stem(t.text.lower()) for t in pdoc if t.is_alpha]
        if term1 in pterms and term2 in pterms:
            return True
    return False


def _wildcard_postings(pattern):
    # convert '!' to '.*' and match against indexed terms (stems)
    regex = '^' + re.escape(pattern).replace(r'\!', '.*') + '$'
    prog = re.compile(regex)
    hits = set()
    for term in term_doc_tf.keys():
        if prog.match(term):
            hits.update(term_doc_tf[term].keys())
    return hits


def extended_boolean_model(raw_query):
    colorizePrint("CYAN", f"[EXTENDED BOOLEAN] Query: {raw_query}")
    tokens = raw_query.strip().split()
    candidates = set(range(N_DOCS))
    optional_scores = defaultdict(int)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Phrase: "...":
        if tok.startswith('"'):
            phrase = tok
            while not phrase.endswith('"') and i + 1 < len(tokens):
                i += 1
                phrase += ' ' + tokens[i]
            phrase = phrase.strip('"')
            matching = {d for d in range(N_DOCS) if _find_phrase_in_doc(phrase, d)}
            candidates &= matching
        elif i + 2 < len(tokens) and re.match(r'^/[\d]+$', tokens[i + 1]):
            term1 = tok
            k = int(tokens[i + 1].lstrip('/'))
            term2 = tokens[i + 2]
            t1 = next((stemmer.stem(t.text.lower()) for t in tokenize(term1) if t.is_alpha), None)
            t2 = next((stemmer.stem(t.text.lower()) for t in tokenize(term2) if t.is_alpha), None)
            if t1 and t2:
                matching = {d for d in range(N_DOCS) if _within_k_words(t1, t2, k, d)}
                candidates &= matching
            i += 2
        elif tok.endswith('/s') or tok.endswith('/p'):
            mode = tok[-2:]
            t1_raw = tok[:-2]
            if i + 1 < len(tokens):
                t2_raw = tokens[i + 1]
                t1 = next((stemmer.stem(t.text.lower()) for t in tokenize(t1_raw) if t.is_alpha), None)
                t2 = next((stemmer.stem(t.text.lower()) for t in tokenize(t2_raw) if t.is_alpha), None)
                if t1 and t2:
                    if mode == '/s':
                        matching = {d for d in range(N_DOCS) if _same_sentence(t1, t2, d)}
                    else:
                        matching = {d for d in range(N_DOCS) if _same_paragraph(t1, t2, d)}
                    candidates &= matching
                i += 1
        elif '!' in tok:
            normalized = tok.replace('*', '!')  # accept '*' as user wildcard too
            matching = _wildcard_postings(normalized)
            candidates &= matching
        else:
            t = next((stemmer.stem(t.text.lower()) for t in tokenize(tok) if t.is_alpha), None)
            if t:
                postings = set(term_doc_tf.get(t, {}).keys())
                for d in postings:
                    optional_scores[d] += 1
        i += 1

    if candidates:
        if optional_scores:
            filtered_scores = {d: s for d, s in optional_scores.items() if d in candidates}
            ranked = sorted(filtered_scores.items(), key=lambda x: (-x[1], x[0]))
            if not ranked:
                ranked = [(d, 0) for d in sorted(candidates)]
        else:
            ranked = [(d, 0) for d in sorted(candidates)]
    else:
        ranked = sorted(optional_scores.items(), key=lambda x: (-x[1], x[0]))

    if not ranked:
        colorizePrint("RED", "No documents matched extended query.")
        return []

    ranked_docs = [d for d, sc in ranked]
    colorizePrint("GREEN", f"Matched documents (ranked): {ranked_docs}")
    return ranked_docs


def vector_space_model(query):
    # todo:
    pass


def probabilistic_model(query):
    # todo:
    pass


def show_standard_boolean_help():
    colorizePrint("YELLOW", "STANDARD BOOLEAN allowed operators:")
    colorizePrint("CYAN", "  AND, OR, NOT, parentheses ( )")
    colorizePrint("CYAN", "  Example: term1 AND (term2 OR term3) AND NOT term4")
    colorizePrint("CYAN", "  Terms are normalized (lowercased) and stemmed.")


def show_extended_boolean_help():
    colorizePrint("YELLOW", "EXTENDED BOOLEAN allowed operators/features:")
    colorizePrint("CYAN", '  "exact phrase"         -> phrase match (use double quotes)')
    colorizePrint("CYAN", "  term /k term           -> within k words (proximity)")
    colorizePrint("CYAN", "  term/s or term/p       -> same sentence (/s) or same paragraph (/p)")
    colorizePrint("CYAN", "  wildcard using '!'     -> e.g. comput! matches computer, computing (rudimentary)")
    colorizePrint("CYAN", "  space-separated terms  -> treated as OR (optional terms scored higher if present)")
    colorizePrint("CYAN", "  Example: \"information retrieval\" /5 search")


def show_vector_help():
    #todo
    pass


def show_probabilistic_help():
    #todo
    pass


def menu():
    print("Choose your retrieval model:\n"
                          "1) Standard Boolean\n"
                          "2) Extended Boolean (Westlaw-like)\n"
                          "3) Vector Space \n"
                          "4) Probabilistic Model\n"
                          "5) Exit")
    while True:
        opt = input("Model (1-5): ").strip()
        try:
            val = int(opt)
            if 1 <= val <= 5:
                return val
        except ValueError:
            pass
        colorizePrint("RED", "Please enter a numeric option (1-5).")


isContinue = True
while isContinue:
    option = menu()
    if option == 5:
        colorizePrint("GREEN", "Goodbye!")
        break

    # show help for chosen model
    if option == 1:
        show_standard_boolean_help()
    elif option == 2:
        show_extended_boolean_help()
    elif option == 3:
        show_vector_help()
    elif option == 4:
        show_probabilistic_help()

    # Ask for the query after model selection
    colorizePrint("YELLOW", "Enter your query (type 'back' to choose another model, 'exit' to quit):")
    query = input().strip()
    if query.lower() == 'exit':
        colorizePrint("GREEN", "Goodbye!")
        break
    if query.lower() == 'back':
        continue

    if option == 1:
        standard_boolean_model(query)
    elif option == 2:
        extended_boolean_model(query)
    elif option == 3:
        vector_space_model(query)
    elif option == 4:
        probabilistic_model(query)
    else:
        colorizePrint("RED", "Invalid option (shouldn't happen).")

    print()