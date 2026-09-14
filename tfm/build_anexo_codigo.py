"""Genera anexo_codigo.html a partir del código fuente real del repo (app/ y
notebooks/barcelona/), y lo renderiza a anexo_codigo.pdf con xhtml2pdf.
Uso: python3 build_anexo_codigo.py
"""
import html
import json
import textwrap
from pathlib import Path

from xhtml2pdf import pisa

ROOT = Path(__file__).parent.parent
TFM_DIR = Path(__file__).parent

HEAD = """<html>
<head>
<meta charset="utf-8">
<style>
    @page {
        size: A4;
        margin: 2.4cm 2.2cm 2.2cm 2.2cm;
        @frame footer_frame {
            -pdf-frame-content: footer_content;
            bottom: 1.1cm; margin-left: 2.2cm; margin-right: 2.2cm; height: 1cm;
        }
    }
    body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.42;
        color: #1a1a1a;
    }
    #footer_content { text-align: center; font-size: 8.5pt; color: #777; }

    h1 { font-size: 16pt; margin-top: 0; margin-bottom: 6pt; color: #111; }
    h2 { font-size: 12pt; margin-top: 16pt; margin-bottom: 4pt; color: #111; }
    h3.filename {
        font-family: Courier, monospace; font-size: 10pt; margin: 14pt 0 4pt 0;
        background: #e8e8e8; padding: 4pt 6pt; color: #111;
    }
    p { margin: 0 0 8pt 0; text-align: justify; }
    ul { margin: 0 0 8pt 0; padding-left: 18pt; }
    li { margin-bottom: 3pt; }

    .codeblock {
        font-family: Courier, monospace;
        font-size: 7.6pt;
        line-height: 1.25;
        word-wrap: break-word;
        background: #f7f7f5;
        border: 0.5pt solid #ccc;
        padding: 6pt 8pt;
        margin: 0 0 12pt 0;
    }

    .portada {
        text-align: center;
        padding-top: 100pt;
        page-break-after: always;
    }
    .portada h1 { font-size: 20pt; margin-bottom: 8pt; }
    .portada p { text-align: center; }

    table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt 0; font-size: 9pt; }
    th, td { border: 0.5pt solid #999; padding: 3.5pt 5pt; text-align: left; vertical-align: top; }
    th { background-color: #e8e8e8; font-weight: bold; }
</style>
</head>
<body>

<div id="footer_content">Anexo III. Página <pdf:pagenumber /> de <pdf:pagecount /></div>

<div class="portada">
    <h1>Anexo III<br>Código completo</h1>
    <p>FairNight: un sistema de recomendación de precios para anuncios de Airbnb</p>
    <p>Trabajo Fin de Máster, Modalidad Semipresencial<br>Yago Coll Crespo, Septiembre 2026</p>
</div>

<h1>Código completo: aplicación FairNight y pipeline de Barcelona</h1>
<p>
Este anexo recoge el código fuente completo de dos partes del proyecto: la aplicación FairNight
(<code>app/</code>) en su totalidad, y el pipeline de modelización de <b>Barcelona</b>
(<code>notebooks/barcelona/</code>, solo celdas de código; el razonamiento y los resultados de cada celda,
en markdown, tablas y gráficas, ya están recogidos en la memoria y en el Anexo II). Barcelona se toma
como ciudad de referencia por ser la más documentada; las otras 6 ciudades ejecutan el <b>mismo pipeline</b>
sobre sus propios datos (Sección 2.4 y 4.5 de la memoria, Anexo I Sección L y Anexo II Sección M detallan en
qué difiere cada una), así que no se repite aquí: su código está disponible en el repositorio público de
GitHub, <a href="https://github.com/yagocoll/Airbnb">github.com/yagocoll/Airbnb</a>.
</p>

<h2>Estructura del repositorio</h2>
__REPO_TREE__

<h2>Dependencias de la aplicación</h2>
<p>Versiones exactas fijadas en <code>app/requirements.txt</code>:</p>
"""

REPO_TREE = """Airbnb/
  app/                        # aplicacion Streamlit FairNight
    app.py                    # interfaz y orquestacion (formulario, prediccion, comparables, mapa)
    features.py               # reconstruye el vector de features que espera cada price_model.pkl
    comparables.py            # seleccion del conjunto de anuncios reales comparables
    requirements.txt          # dependencias exactas de la app
    assets/<ciudad>/          # geojson de distritos, importancia de variables, fotos, logo
  notebooks/<ciudad>/         # 01_eda, 02_feature_engineering, 03_model_baseline,
                               # 04_model_training, 05_model_evaluation (una carpeta por ciudad)
  models/<ciudad>/            # price_model.pkl + model_metadata.json (features, hiperparametros)
  data/
    raw/<ciudad>/             # listings.csv.gz de Inside Airbnb (no versionado, pesa demasiado)
    processed/<ciudad>/       # csv limpios y con features, generados por los notebooks
  tfm/                         # memoria.html/pdf y los tres anexos (variables, EDA, codigo)
"""

FOOTER = """
</body>
</html>
"""


# Ancho de columna en caracteres del bloque de codigo: pagina A4 (595.28pt) menos
# margenes (2.2cm a cada lado) menos el padding del propio .codeblock (8pt a cada lado),
# dividido por el ancho de caracter de Courier a 7.6pt (exactamente 0.6 * tamano de fuente,
# por ser monoespaciada). Con margen de seguridad, redondeado hacia abajo.
MAX_CHARS = 97


def _wrap_line(line: str) -> list[str]:
    stripped = line.lstrip(" ")
    n_lead = len(line) - len(stripped)
    if not stripped:
        return ["&nbsp;"]
    avail = max(MAX_CHARS - n_lead, 20)
    if len(stripped) <= avail:
        return ["&nbsp;" * n_lead + html.escape(stripped)]
    # xhtml2pdf no envuelve de forma fiable lineas largas (ni con white-space:pre-wrap
    # ni con word-wrap:break-word: se comprobo que una cadena larga sin espacios se sale
    # de la pagina en vez de partirse), asi que el ajuste de linea se hace aqui a mano,
    # con la misma sangria del codigo original en cada continuacion.
    chunks = textwrap.wrap(stripped, width=avail, break_long_words=True, break_on_hyphens=False)
    return ["&nbsp;" * n_lead + html.escape(chunk) for chunk in chunks]


def code_html(text: str) -> str:
    # xhtml2pdf tampoco respeta saltos de linea reales dentro de <pre>/white-space:pre-wrap
    # (los colapsa igual que en texto normal): cada linea de codigo se separa a mano con
    # <br/> en un unico bloque de texto (usar un <div> por linea hace que xhtml2pdf repita
    # el fondo/borde del contenedor en cada linea, multiplicando el numero de paginas).
    out_lines = []
    for line in text.split("\n"):
        out_lines.extend(_wrap_line(line))
    return f'<div class="codeblock">{"<br/>".join(out_lines)}</div>'


def build_requirements_table():
    req_path = ROOT / "app" / "requirements.txt"
    rows = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        pkg, _, version = line.partition("==")
        rows.append(f"<tr><td><code>{pkg}</code></td><td>{version}</td></tr>")
    return (
        "<table><tr><th>Paquete</th><th>Versión</th></tr>" + "".join(rows) + "</table>"
    )


def build_app_section():
    parts = ["<h1>Aplicación FairNight</h1>"]
    files = [
        ("app/app.py", "Interfaz Streamlit completa: formulario de descripción del anuncio, "
         "predicción, explicación (SHAP), comparables y mapa."),
        ("app/features.py", "Reconstruye el vector de features que espera price_model.pkl a partir "
         "de las respuestas del formulario."),
        ("app/comparables.py", "Selección de un conjunto único de anuncios reales comparables, "
         "consumido por las pantallas de veredicto, lista y mapa."),
    ]
    for rel_path, desc in files:
        parts.append(f'<h3 class="filename">{rel_path}</h3>')
        parts.append(f"<p>{desc}</p>")
        content = (ROOT / rel_path).read_text()
        parts.append(code_html(content))
    return "\n".join(parts)


def extract_notebook_code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    chunks = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"]).rstrip()
            if src:
                chunks.append(src)
    return "\n\n".join(chunks)


def build_notebooks_section():
    parts = ["<h1>Pipeline de modelización (Barcelona)</h1>"]
    notebooks = [
        ("01_eda.ipynb", "Carga, tipado, nulos, duplicados, análisis univariante/bivariante, "
         "correlaciones, geografía y amenities. Detalle narrativo completo en el Anexo II."),
        ("02_feature_engineering.ipynb", "Construcción de las 71 columnas de features del modelo "
         "final a partir de las 43 columnas de partida (Sección 3 de la memoria)."),
        ("03_model_baseline.ipynb", "Baselines (media, mediana, mediana por bedrooms, regresión "
         "lineal) y detección/corrección de la fuga en neighbourhood_price_encoded (Sección 3.4)."),
        ("04_model_training.ipynb", "Comparativa de las cuatro familias de modelos y ajuste de "
         "hiperparámetros con RandomizedSearchCV (Sección 4.2-4.3)."),
        ("05_model_evaluation.ipynb", "Evaluación final en test, importancia de variables y SHAP "
         "(Secciones 4.4 y 5 de la memoria)."),
    ]
    for fname, desc in notebooks:
        parts.append(f'<h3 class="filename">notebooks/barcelona/{fname}</h3>')
        parts.append(f"<p>{desc}</p>")
        code = extract_notebook_code(ROOT / "notebooks" / "barcelona" / fname)
        parts.append(code_html(code))
    return "\n".join(parts)


def main():
    head = HEAD.replace("__REPO_TREE__", code_html(REPO_TREE))
    html_out = head + build_requirements_table() + build_app_section() + build_notebooks_section() + FOOTER
    src = TFM_DIR / "anexo_codigo.html"
    dest = TFM_DIR / "anexo_codigo.pdf"
    src.write_text(html_out, encoding="utf-8")

    with open(dest, "wb") as f_out:
        result = pisa.CreatePDF(html_out, dest=f_out, encoding="utf-8", path=str(TFM_DIR) + "/")
    if result.err:
        print(f"ERRORES: {result.err}")
    else:
        print(f"OK -> {dest}")


if __name__ == "__main__":
    main()
