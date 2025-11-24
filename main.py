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
    lemma_terms = set(
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
    )
    return lemma_terms


def build_inverted_index(list_of_lemma_sets):
    btree_index = OOBTree()
    for doc_id, lemma_sets in enumerate(list_of_lemma_sets):
        for lemma in lemma_sets:
            if lemma not in btree_index:
                btree_index[lemma] = set()
            btree_index[lemma].add(doc_id)
    return btree_index


def print_colored(title, content, color=Colors.GREEN):
    """Helper function to print colored titles"""
    print(f"{Colors.BOLD}{color}{title}{Colors.RESET}")
    print(content)


# Example usage for multiple PDFs
pdf_files = ['./networking.pdf', './OWASP.pdf']
lemma_sets = []

print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 50}{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}Processing PDF Documents{Colors.RESET}")
print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 50}{Colors.RESET}\n")

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"{Colors.CYAN}Processing Document {idx}: {pdf_path}{Colors.RESET}")
    print(f"{Colors.YELLOW}Tokenizing docs process start.{Colors.RESET}")

    text = extract_text_from_pdf(pdf_path)
    doc = tokenize(text)

    print(f"{Colors.GREEN}List of words:{Colors.RESET}")
    words_list = [token.text for token in doc if token.is_alpha]
    print(f"  {words_list[:20]}...")  # Show first 20 words

    lemma_terms = lemmatize(doc)
    lemma_sets.append(lemma_terms)

    print(f"{Colors.BLUE}List of lemmatized words (terms):{Colors.RESET}")
    print(f"  {sorted(list(lemma_terms))[:20]}...")  # Show first 20 lemmas
    print()

print(f"{Colors.BOLD}{Colors.YELLOW}Creating inverted index start{Colors.RESET}")
btree_index = build_inverted_index(lemma_sets)

print(f"{Colors.BOLD}{Colors.CYAN}Inverted index created successfully!{Colors.RESET}")
print(f"{Colors.CYAN}Total unique terms: {len(btree_index)}{Colors.RESET}\n")

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
        result = btree_index.get(query.lower(), set())
        print(f"{Colors.BOLD}{Colors.CYAN}Documents containing \"{query}\":{Colors.RESET}")
        if result:
            print(f"{Colors.GREEN}  Found in documents: {sorted(list(result))}{Colors.RESET}")
        else:
            print(f"{Colors.RED}  No documents found containing this term.{Colors.RESET}")
        print()