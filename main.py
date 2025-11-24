import pdfplumber
import spacy
from BTrees.OOBTree import OOBTree

nlp = spacy.load("en_core_web_sm")

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
    lemma_terms = set(
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
    )
    return lemma_terms

def build_inverted_index(list_of_lemma_sets):
    btree_index = OOBTree()
    for doc_id, lemmas in enumerate(list_of_lemma_sets):
        for lemma in lemmas:
            if lemma not in btree_index:
                btree_index[lemma] = set()
            btree_index[lemma].add(doc_id)
    return btree_index

# Example usage for multiple PDFs
pdf_files = ['./Eureka_02_03_2025_.pdf', './networking.pdf']
lemma_sets = []
for pdf_path in pdf_files:
    text = extract_text_from_pdf(pdf_path)
    doc = tokenize(text)
    lemma_terms = lemmatize(doc)
    lemma_sets.append(lemma_terms)

btree_index = build_inverted_index(lemma_sets)

query = input("Enter a term to search: ")
# Example: access documents containing the lemma "network"
print("Documents containing 'network':", btree_index.get('network', set()))
print(f'Documents containing "{query}":', btree_index.get(query, set()))



