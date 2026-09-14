"""Genera memoria.pdf a partir de memoria.html. Uso: python3 build.py"""
from pathlib import Path
from xhtml2pdf import pisa

DIR = Path(__file__).parent
src = DIR / "memoria.html"
dest = DIR / "memoria.pdf"

with open(src, encoding="utf-8") as f_in, open(dest, "wb") as f_out:
    # path=str(DIR): las <img src="assets/..."> del HTML son relativas a tfm/, no al cwd
    # desde el que se lance este script.
    result = pisa.CreatePDF(f_in.read(), dest=f_out, encoding="utf-8", path=str(DIR) + "/")

if result.err:
    print(f"ERRORES: {result.err}")
else:
    print(f"OK -> {dest}")
