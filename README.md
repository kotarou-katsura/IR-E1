#   Inverted Index  and B-tree Builder

<img width="1745" height="801" alt="image" src="https://github.com/user-attachments/assets/3e19a149-7cb0-4d27-855b-3b5ced330300" />
<img width="1523" height="799" alt="image" src="https://github.com/user-attachments/assets/3ee37b40-c398-4dba-9908-d682f683ea77" />


Before running the script, make sure the required libraries are installed:

```
pip install pdfplumber spacy BTrees
python -m spacy download en_core_web_sm
```
The project uses three sample PDFs to demonstrate how indexing works:

```pdf_files = ['./ow07.pdf', './networking.pdf', './ow04.pdf']```


# Retrieval Models

**STANDARD BOOLEAN**  
Allowed operators: `AND`, `OR`, `NOT`, parentheses `( )`  

**EXTENDED BOOLEAN**  
Allowed operators/features:  
- `"exact phrase"` → phrase match (use double quotes)  
- `term /k term` → within *k* words (proximity)  
- `term/s` or `term/p` → same sentence (`/s`) or same paragraph (`/p`)  
- Wildcard using `!` → e.g., `comput!` matches `computer`, `computing`  
- Space-separated terms → treated as OR (optional terms scored higher if present)  


  
