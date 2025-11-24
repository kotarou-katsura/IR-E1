import pdfplumber
import spacy
from BTrees.OOBTree import OOBTree

nlp = spacy.load("en_core_web_sm")


# ANSI Color codes
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
    # Keep original text capitalization as separate terms
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
            # Keep original capitalization (case-sensitive)
            key = lemma
            if key not in inverted_index:
                inverted_index[key] = {
                    'frequency': 0,
                    'postings': set()  # Document IDs
                }
            inverted_index[key]['frequency'] += 1
            inverted_index[key]['postings'].add(doc_id)

    return inverted_index


# Example usage for multiple PDFs
pdf_files = ['./ow07.pdf', './networking.pdf', './ow04.pdf']
lemma_sets = []

print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 50}{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}Processing PDF Documents{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 50}{Colors.RESET}\n")

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"{Colors.CYAN}Processing Document {idx}: {pdf_path}{Colors.RESET}")
    print(f"{Colors.YELLOW}Tokenizing ...{Colors.RESET}")

    text = extract_text_from_pdf(pdf_path)
    doc = tokenize(text)

    print(f"{Colors.GREEN}List of words:{Colors.RESET}")
    words_list = [token.text for token in doc if token.is_alpha]
    print(f"  {words_list[:100]}...")  # Show first 20 words

    lemma_terms = lemmatize(doc)
    lemma_sets.append(lemma_terms)

    print(f"{Colors.BLUE}List of lemmatized words (terms):{Colors.RESET}")
    # Sort and display lemmatized terms with original capitalization
    lemma_list = sorted(list(set(lemma_terms)), key=str.lower)
    print(f"  {lemma_list[:100]}...")  # Show first 20 lemmas
    print()

print(f"{Colors.GREEN}=================================={Colors.RESET}")
print(f"{Colors.BOLD}{Colors.YELLOW}Creating B-Tree for documents start{Colors.RESET}")
doc_btree = build_document_btree(pdf_files, lemma_sets)
print(f"{Colors.BOLD}{Colors.CYAN}Document B-Tree created successfully!{Colors.RESET}")
print(f"{Colors.CYAN}Total documents stored: {len(doc_btree)}{Colors.RESET}\n")

print(f"{Colors.BOLD}{Colors.YELLOW}Creating inverted index start{Colors.RESET}")
inverted_index = build_inverted_index(lemma_sets)
print(f"{Colors.BOLD}{Colors.CYAN}Inverted index created successfully!{Colors.RESET}")
print(f"{Colors.CYAN}Total unique terms: {len(inverted_index)}{Colors.RESET}\n")

print(f"{Colors.BLUE}Inverted Index (Term | Frequency | Postings):{Colors.RESET}")
print(f"{Colors.YELLOW}{'Term':<30} | {'Frequency':<12} | {'Postings':<20}{Colors.RESET}")
print(f"{Colors.YELLOW}{'-'*30}-+-{'-'*12}-+-{'-'*20}{Colors.RESET}")
# Show ALL terms from inverted index
all_terms = sorted(list(inverted_index.keys()))
for term in all_terms:
    frequency = inverted_index[term]['frequency']
    postings = sorted(list(inverted_index[term]['postings']))
    print(f"{Colors.CYAN}{term:<30}{Colors.RESET} | {frequency:<12} | {str(postings):<20}")
print()

# Search loop
print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 50}{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}Search Section{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 50}{Colors.RESET}\n")

isContinue = True
while isContinue:
    query = input(f"{Colors.BOLD}{Colors.YELLOW}Enter a term to search (type 'exit' to quit): {Colors.RESET}")

    if query.lower() == 'exit':
        isContinue = False
        print(f"{Colors.BOLD}{Colors.GREEN}Goodbye!{Colors.RESET}")
    else:
        # Case-sensitive search
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