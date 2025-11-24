# PDF Lemmatization & Inverted Index Builder

This project extracts text from PDF documents, tokenizes and lemmatizes the content using **spaCy**, and builds an **inverted index** using a B-Tree structure (`OOBTree`).  
The index allows fast searching of terms across multiple PDF files.

---

## Requirements

Before running the script, make sure the required libraries are installed:

```python
import pdfplumber
import spacy
from BTrees.OOBTree import OOBTree
```

Sample PDF Files
The project uses three sample PDFs to demonstrate how indexing works:

```pdf_files = ['./ow07.pdf', './networking.pdf', './ow04.pdf']```
