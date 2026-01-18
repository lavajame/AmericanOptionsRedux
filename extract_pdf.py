
import sys
import os

def extract_text(pdf_path, output_path):
    text = ""
    try:
        import pypdf
        print(f"Using pypdf for {pdf_path}")
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except ImportError:
        try:
            import PyPDF2
            print(f"Using PyPDF2 for {pdf_path}")
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfFileReader(f)
                for i in range(reader.numPages):
                    text += reader.getPage(i).extractText() + "\n"
        except ImportError:
            print("No suitable PDF library found (pypdf or PyPDF2).")
            return False

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return True

if __name__ == "__main__":
    files = ["Ju_1999.pdf", "ssrn-362.pdf"]
    for f in files:
        if os.path.exists(f):
            out = f.replace(".pdf", ".txt")
            if extract_text(f, out):
                print(f"Extracted {f} to {out}")
            else:
                print(f"Failed to extract {f}")
        else:
            print(f"File {f} not found")
