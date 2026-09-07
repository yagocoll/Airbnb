import json
from pathlib import Path

import altair as alt
import branca.colormap as cm
import folium
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import shap
import streamlit as st
from streamlit_folium import st_folium

from features import (
    DISTRICTS,
    FEATURE_COLS,
    METADATA,
    NEIGHBOURHOODS_BY_DISTRICT,
    NEIGHBOURHOOD_STATS,
    PROPERTY_TYPES,
    ROOM_TYPES,
    build_features,
    confidence_flags,
    host_size_tier,
    pretty_feature_name,
)

ROOM_TYPE_ICONS = {
    "Entire home/apt": ":material/home: Vivienda entera",
    "Private room": ":material/door_front: Habitación privada",
    "Shared room": ":material/group: Habitación compartida",
    "Hotel room": ":material/hotel: Habitación de hotel",
}
ROOM_TYPE_LABELS = {v: k for k, v in ROOM_TYPE_ICONS.items()}

REVIEW_DEFAULTS = {
    "number_of_reviews": 10,
    "number_of_reviews_ltm": 5,
    "reviews_per_month": 1.0,
    "review_scores_rating": 4.8,
    "listing_age_days": 365,
    "days_since_last_review": 30,
    "estimated_occupancy_l365d": 100,
}

WHATIF_LABELS = {
    "district": "Distrito",
    "neighbourhood": "Barrio",
    "room_type": "Tipo de alojamiento",
    "property_type": "Tipo de propiedad",
    "accommodates": "Huéspedes",
    "bedrooms": "Dormitorios",
    "beds": "Camas",
    "bathrooms": "Baños",
    "minimum_nights": "Estancia mínima (noches)",
    "maximum_nights": "Estancia máxima (noches)",
    "availability_365": "Días disponibles al año",
    "has_license": "Tiene licencia turística",
    "host_is_superhost": "Soy superhost",
    "host_has_profile_pic": "Tengo foto de perfil",
    "host_identity_verified": "Identidad verificada",
    "calculated_host_listings_count": "Anuncios que gestiono",
    "host_tenure_years": "Años como anfitrión",
    "host_user_tenure_years": "Años con cuenta en la plataforma",
    "is_new_listing": "Es un anuncio nuevo (sin reseñas)",
    "number_of_reviews": "Número de reseñas",
    "number_of_reviews_ltm": "Reseñas último año",
    "reviews_per_month": "Reseñas al mes",
    "review_scores_rating": "Valoración media",
    "listing_age_days": "Días desde la publicación",
    "days_since_last_review": "Días desde la última reseña",
    "estimated_occupancy_l365d": "Noches ocupadas último año",
}

WHATIF_NUMERIC_VALUES = {
    "accommodates": [1, 2, 4, 6, 8, 10, 12, 14, 16],
    "bedrooms": [0, 1, 2, 3, 4, 6, 8, 10],
    "beds": [1, 2, 3, 4, 6, 8, 10],
    "bathrooms": [0.5, 1, 1.5, 2, 3, 4, 6, 8],
    "minimum_nights": [1, 2, 3, 7, 14, 30, 60, 90, 180, 365],
    "maximum_nights": [7, 30, 90, 180, 365, 730, 1125],
    "availability_365": [0, 30, 90, 180, 270, 365],
    "calculated_host_listings_count": [1, 2, 5, 10, 25, 50, 100, 250, 500],
    "host_tenure_years": [0, 0.5, 1, 2, 5, 10, 15, 20],
    "host_user_tenure_years": [0.5, 1, 2, 5, 10, 15, 20, 25],
    "number_of_reviews": [0, 1, 5, 10, 25, 50, 100, 250],
    "number_of_reviews_ltm": [0, 1, 5, 10, 25, 50],
    "reviews_per_month": [0, 0.5, 1, 2, 4, 8],
    "review_scores_rating": [3.0, 3.5, 4.0, 4.5, 4.8, 5.0],
    "listing_age_days": [30, 90, 180, 365, 730, 1460, 2920],
    "days_since_last_review": [0, 7, 30, 90, 180, 365],
    "estimated_occupancy_l365d": [0, 10, 30, 60, 100, 150, 250, 365],
    "neighbourhood_price_encoded": [80, 120, 160, 200, 250, 300, 350, 400, 450],
    "distance_to_center_km": [0, 0.5, 1, 2, 3, 4, 5, 6, 8],
    "host_entire_homes_ratio": [0, 0.25, 0.5, 0.75, 1.0],
}

PDP_GROUPS = {
    "Estancia mínima": {"kind": "numeric", "cols": ["minimum_nights"], "sample_key": "minimum_nights"},
    "Huéspedes": {"kind": "numeric", "cols": ["accommodates"], "sample_key": "accommodates"},
    "Tipo de alojamiento": {"kind": "onehot", "prefix": "room_type_", "options": ROOM_TYPES},
    "Baños": {"kind": "numeric", "cols": ["bathrooms", "bathrooms_log"], "sample_key": "bathrooms"},
    "Dormitorios": {"kind": "numeric", "cols": ["bedrooms", "bedrooms_log"], "sample_key": "bedrooms"},
    "Anuncios que gestiona": {"kind": "host_scale"},
    "Valoración media": {"kind": "numeric", "cols": ["review_scores_rating"], "sample_key": "review_scores_rating"},
    "Días disponibles al año": {"kind": "numeric", "cols": ["availability_365"], "sample_key": "availability_365"},
    "Nivel de precios del barrio": {"kind": "numeric", "cols": ["neighbourhood_price_encoded"], "sample_key": "neighbourhood_price_encoded"},
    "Tipo de propiedad": {"kind": "onehot", "prefix": "property_type_", "options": PROPERTY_TYPES},
    "Años como anfitrión": {"kind": "numeric", "cols": ["host_tenure_years"], "sample_key": "host_tenure_years"},
    "Años con cuenta en la plataforma": {"kind": "numeric", "cols": ["host_user_tenure_years"], "sample_key": "host_user_tenure_years"},
    "Días desde la última reseña": {"kind": "numeric", "cols": ["days_since_last_review"], "sample_key": "days_since_last_review"},
    "Estancia máxima": {"kind": "numeric", "cols": ["maximum_nights"], "sample_key": "maximum_nights"},
    "Licencia turística": {"kind": "boolean", "col": "has_license"},
    "Distancia al centro": {"kind": "numeric", "cols": ["distance_to_center_km"], "sample_key": "distance_to_center_km"},
    "Reseñas al mes": {"kind": "numeric", "cols": ["reviews_per_month", "reviews_per_month_log"], "sample_key": "reviews_per_month"},
    "Antigüedad del anuncio": {"kind": "numeric", "cols": ["listing_age_days"], "sample_key": "listing_age_days"},
    "Superhost": {"kind": "boolean", "col": "host_is_superhost"},
    "Número de reseñas": {"kind": "numeric", "cols": ["number_of_reviews", "number_of_reviews_log"], "sample_key": "number_of_reviews"},
    "Reseñas último año": {"kind": "numeric", "cols": ["number_of_reviews_ltm", "number_of_reviews_ltm_log"], "sample_key": "number_of_reviews_ltm"},
    "Noches ocupadas último año": {"kind": "numeric", "cols": ["estimated_occupancy_l365d"], "sample_key": "estimated_occupancy_l365d"},
    "Camas": {"kind": "numeric", "cols": ["beds", "beds_log"], "sample_key": "beds"},
    "Distrito": {"kind": "onehot", "prefix": "district_", "options": DISTRICTS},
    "Identidad verificada": {"kind": "boolean", "col": "host_identity_verified"},
    "Tiene reseñas": {"kind": "boolean", "col": "has_reviews"},
    "Foto de perfil": {"kind": "boolean", "col": "host_has_profile_pic"},
    "% de vivienda entera del anfitrión": {"kind": "numeric", "cols": ["host_entire_homes_ratio"], "sample_key": "host_entire_homes_ratio"},
}

# De cada clave de WHATIF_LABELS a la fila del waterfall que le corresponde (las
# etiquetas de PDP_GROUPS), para poder resaltar en "Por qué este precio" justo la
# variable que se está simulando en "¿Qué pasaría si cambio...?".
WHATIF_KEY_TO_GROUP = {
    "district": "Distrito",
    "neighbourhood": "Nivel de precios del barrio",
    "room_type": "Tipo de alojamiento",
    "property_type": "Tipo de propiedad",
    "accommodates": "Huéspedes",
    "bedrooms": "Dormitorios",
    "beds": "Camas",
    "bathrooms": "Baños",
    "minimum_nights": "Estancia mínima",
    "maximum_nights": "Estancia máxima",
    "availability_365": "Días disponibles al año",
    "has_license": "Licencia turística",
    "host_is_superhost": "Superhost",
    "host_has_profile_pic": "Foto de perfil",
    "host_identity_verified": "Identidad verificada",
    "calculated_host_listings_count": "Anuncios que gestiona",
    "host_tenure_years": "Años como anfitrión",
    "host_user_tenure_years": "Años con cuenta en la plataforma",
    "is_new_listing": "Tiene reseñas",
    "number_of_reviews": "Número de reseñas",
    "number_of_reviews_ltm": "Reseñas último año",
    "reviews_per_month": "Reseñas al mes",
    "review_scores_rating": "Valoración media",
    "listing_age_days": "Antigüedad del anuncio",
    "days_since_last_review": "Días desde la última reseña",
    "estimated_occupancy_l365d": "Noches ocupadas último año",
}

# Para el resumen SHAP: solo variables con un valor propio y continuo (numéricas y
# booleanas). Las agrupadas en one-hot (distrito, tipo...) no tienen un "valor" único
# que colorear y ya se ven en el ranking de importancia y en el PDP categórico.
SHAP_SUMMARY_GROUPS = {
    label: spec for label, spec in PDP_GROUPS.items() if spec["kind"] in ("numeric", "boolean")
}

# De cada una de las 66 columnas crudas del modelo a la variable "de negocio" a la que
# pertenece (p.ej. las 10 columnas district_* se agrupan todas bajo "Distrito"), para no
# mostrar nunca una categoría que el usuario no ha elegido (Distrito: Gràcia, etc.).
_HOST_SCALE_COLS = [
    "calculated_host_listings_count", "calculated_host_listings_count_log",
    "host_size_individual", "host_size_pequeno", "host_size_mediano", "host_size_grande",
]
FEATURE_TO_GROUP = {}
for _label, _spec in PDP_GROUPS.items():
    if _spec["kind"] == "numeric":
        for _c in _spec["cols"]:
            FEATURE_TO_GROUP[_c] = _label
    elif _spec["kind"] == "boolean":
        FEATURE_TO_GROUP[_spec["col"]] = _label
    elif _spec["kind"] == "onehot":
        for _c in FEATURE_COLS:
            if _c.startswith(_spec["prefix"]):
                FEATURE_TO_GROUP[_c] = _label
    elif _spec["kind"] == "host_scale":
        for _c in _HOST_SCALE_COLS:
            FEATURE_TO_GROUP[_c] = _label
for _c in FEATURE_COLS:
    FEATURE_TO_GROUP.setdefault(_c, pretty_feature_name(_c))


def pdp_apply(Xc, group_label, value):
    spec = PDP_GROUPS[group_label]
    kind = spec["kind"]
    if kind == "numeric":
        for col in spec["cols"]:
            Xc[col] = np.log1p(value) if col.endswith("_log") else value
    elif kind == "boolean":
        Xc[spec["col"]] = int(value)
    elif kind == "onehot":
        onehot_cols = [c for c in Xc.columns if c.startswith(spec["prefix"])]
        Xc[onehot_cols] = 0
        col = f"{spec['prefix']}{value}"
        if col in Xc.columns:
            Xc[col] = 1
    elif kind == "host_scale":
        Xc["calculated_host_listings_count"] = value
        Xc["calculated_host_listings_count_log"] = np.log1p(value)
        tier = host_size_tier(value)
        for t in ["individual", "pequeno", "mediano", "grande"]:
            Xc[f"host_size_{t}"] = 1 if t == tier else 0
    else:
        raise ValueError(f"kind desconocido: {kind}")
    return Xc


def pdp_axis_values(group_label):
    spec = PDP_GROUPS[group_label]
    kind = spec["kind"]
    if kind == "numeric":
        return WHATIF_NUMERIC_VALUES[spec["sample_key"]], "numeric"
    if kind == "host_scale":
        return WHATIF_NUMERIC_VALUES["calculated_host_listings_count"], "numeric"
    if kind == "boolean":
        return [False, True], "categorical"
    if kind == "onehot":
        return spec["options"], "categorical"
    raise ValueError(f"kind desconocido: {kind}")


def pdp_label(value):
    if value is True:
        return "Sí"
    if value is False:
        return "No"
    return str(value)


def pdp_real_values(X_base, group_label):
    """Valores reales (no los puntos curados de muestreo) de una variable numérica
    en el test set, para dibujar su histograma de soporte bajo el PDP."""
    spec = PDP_GROUPS[group_label]
    if spec["kind"] == "numeric":
        return X_base[spec["sample_key"]]
    if spec["kind"] == "host_scale":
        return X_base["calculated_host_listings_count"]
    raise ValueError(f"pdp_real_values no aplica a variables categóricas: {group_label}")


def pdp_correlation_column(group_label):
    """Columna cruda para calcular correlación real, si la variable tiene una sola
    (numérica, booleana o el conteo de anuncios del anfitrión). None para las
    agrupadas en one-hot (distrito, tipo...), que no tienen un único valor a correlacionar."""
    spec = PDP_GROUPS[group_label]
    if spec["kind"] == "numeric":
        return spec["sample_key"]
    if spec["kind"] == "boolean":
        return spec["col"]
    if spec["kind"] == "host_scale":
        return "calculated_host_listings_count"
    return None


def pdp_pair_correlation(X_base, label_a, label_b):
    col_a = pdp_correlation_column(label_a)
    col_b = pdp_correlation_column(label_b)
    if col_a is None or col_b is None:
        return np.nan
    return float(X_base[col_a].astype(float).corr(X_base[col_b].astype(float)))


def interaction_diagnosis(indice, corr):
    """Explicación en una frase de si conviene fiarse del índice de interacción de una
    pareja, cruzándolo con la correlación real entre ambas variables."""
    if indice < 0.10:
        return "Apenas interactúan: cada variable actúa casi de forma independiente."
    nivel = "fuerte" if indice > 0.25 else "moderada"
    if pd.isna(corr):
        return f"Interacción {nivel}, pero no se puede comprobar si es artefacto: alguna es una variable agrupada."
    if abs(corr) >= 0.5:
        return f"Ambigua: correlación real alta (r={corr:.2f}) entre ambas, gran parte del índice puede venir de combinaciones poco realistas."
    if abs(corr) >= 0.25:
        return f"Parcialmente ambigua: hay algo de correlación real (r={corr:.2f}) que puede inflar el índice."
    return f"Interacción {nivel} y probablemente genuina: poca correlación real entre ambas (r={corr:.2f})."


# Hipótesis de negocio curadas a mano para el top-15 de interaction_scores.json: por qué
# tendría sentido de negocio que estas dos variables se combinen, más allá del índice y la
# correlación. Cuando no encuentro una historia convincente, lo digo explícitamente en vez
# de forzar una explicación (puede ser simplemente ruido del modelo en zonas con pocos datos).
INTERACTION_BUSINESS_NOTES = {
    frozenset({"Anuncios que gestiona", "Días disponibles al año"}):
        "Los anfitriones profesionales (muchos anuncios) suelen mantener sus pisos disponibles casi todo el "
        "año como negocio a tiempo completo; uno con un solo piso puede tenerlo disponible solo parte del año.",
    frozenset({"Años como anfitrión", "Años con cuenta en la plataforma"}):
        "No hay una historia de negocio real detrás: son casi la misma variable (quien lleva más tiempo como "
        "anfitrión también lleva más tiempo con cuenta), así que el índice alto es sobre todo redundancia.",
    frozenset({"Baños", "Reseñas último año"}):
        "No encuentro una razón de negocio clara. Con tan poca correlación entre ambas, y pocos anuncios reales "
        "con muchos baños y muchas reseñas recientes a la vez, esto puede ser ruido del modelo, no un patrón real.",
    frozenset({"Estancia máxima", "Antigüedad del anuncio"}):
        "Los anfitriones con anuncios más antiguos han tenido más tiempo para ajustar su política de estancia "
        "máxima a lo que realmente funciona; en anuncios nuevos ese valor suele ser más genérico.",
    frozenset({"Estancia máxima", "Reseñas al mes"}):
        "Una estancia máxima corta favorece más rotación de huéspedes y por tanto más reseñas por mes: el "
        "efecto de la estancia máxima sobre el precio podría depender de cuánta rotación real tiene el anuncio.",
    frozenset({"Tipo de alojamiento", "Baños"}):
        "Tiene sentido de negocio: más baños importa mucho en una vivienda entera (todos los huéspedes los "
        "disfrutan), pero apenas cambia la experiencia en una habitación privada o compartida.",
    frozenset({"Años con cuenta en la plataforma", "Distancia al centro"}):
        "No encuentro una historia de negocio convincente; con correlación casi nula, si hay algo real aquí "
        "probablemente sea ruido de combinaciones poco pobladas (cuentas muy antiguas lejos del centro).",
    frozenset({"Años con cuenta en la plataforma", "Estancia máxima"}):
        "Especulativo: podría reflejar que Airbnb ha cambiado sus políticas y valores por defecto de estancia "
        "máxima a lo largo de los años, y los anfitriones más veteranos configuraron el suyo en otro contexto.",
    frozenset({"Años con cuenta en la plataforma", "Superhost"}):
        "Tiene sentido: ser Superhost exige historial demostrado, así que muchos años en la plataforma más el "
        "sello de Superhost transmiten más confianza conjunta que cualquiera de las dos por separado.",
    frozenset({"Días disponibles al año", "Distancia al centro"}):
        "Plausible: en el centro, alta disponibilidad todo el año suele señalar un piso turístico profesional "
        "(más caro); en la periferia, alta disponibilidad puede indicar simplemente poca demanda.",
    frozenset({"Antigüedad del anuncio", "Distrito"}):
        "Los distritos turísticos más consolidados llevan más tiempo con oferta activa en Airbnb que zonas más "
        "residenciales o de incorporación más reciente al mercado.",
    frozenset({"Estancia mínima", "Huéspedes"}):
        "Coherente con lo visto en el mapa: los anuncios para muchos huéspedes suelen ser alquiler de temporada "
        "(estancias largas), mientras que los pequeños son turismo de estancia corta.",
    frozenset({"Huéspedes", "Dormitorios"}):
        "Poca sorpresa de negocio: más dormitorios casi siempre implica más capacidad, así que gran parte del "
        "índice es solo esa correlación natural entre ambas, no una interacción genuina.",
    frozenset({"Huéspedes", "Tipo de alojamiento"}):
        "Tiene sentido: el número de huéspedes importa mucho en vivienda entera (caben grupos grandes) pero "
        "apenas varía en habitación privada o compartida, donde casi siempre son 1-2 personas.",
    frozenset({"Antigüedad del anuncio", "Noches ocupadas último año"}):
        "Plausible: un anuncio más antiguo ha tenido más tiempo para ganar visibilidad y reseñas, lo que puede "
        "cambiar cómo se traduce la ocupación reciente en precio frente a uno recién publicado.",
}


def interaction_business_note(label_a, label_b):
    return INTERACTION_BUSINESS_NOTES.get(
        frozenset({label_a, label_b}),
        "Sin hipótesis de negocio guardada para esta pareja; interpreta el índice y la correlación de arriba.",
    )


def pdp_category_counts(X_base, group_label, raw_values):
    """Nº de anuncios reales del test set en cada categoría, para variables categóricas."""
    spec = PDP_GROUPS[group_label]
    if spec["kind"] == "onehot":
        return [int(X_base[f"{spec['prefix']}{v}"].sum()) for v in raw_values]
    if spec["kind"] == "boolean":
        col = X_base[spec["col"]]
        return [int((col == int(v)).sum()) for v in raw_values]
    raise ValueError(f"pdp_category_counts no aplica a variables numéricas: {group_label}")


def pdp_curve(model, X_base, group_label):
    values, kind = pdp_axis_values(group_label)
    results = [float(np.expm1(model.predict(pdp_apply(X_base.copy(), group_label, v))).mean()) for v in values]
    labels = values if kind == "numeric" else [pdp_label(v) for v in values]
    return labels, results, kind


def pdp_surface(model, X_base, group_a, group_b):
    values_a, kind_a = pdp_axis_values(group_a)
    values_b, kind_b = pdp_axis_values(group_b)
    rows = []
    for va in values_a:
        Xa = pdp_apply(X_base.copy(), group_a, va)
        for vb in values_b:
            Xc = pdp_apply(Xa.copy(), group_b, vb)
            price = float(np.expm1(model.predict(Xc)).mean())
            rows.append({"a": pdp_label(va), "b": pdp_label(vb), "a_raw": va, "b_raw": vb, "price": price})
    df = pd.DataFrame(rows)
    order_a = df.drop_duplicates("a").sort_values("a_raw")["a"].tolist() if kind_a == "numeric" else sorted(df["a"].unique())
    order_b = df.drop_duplicates("b").sort_values("b_raw")["b"].tolist() if kind_b == "numeric" else sorted(df["b"].unique())
    return df, order_a, order_b


def interaction_strength(grid_df):
    """Compara la superficie real con la que habría si las dos variables actuaran de forma
    independiente (efecto de cada una por separado, sin combinarse). Cuanto más se aleje la
    superficie real de esa versión aditiva, más fuerte es la interacción."""
    pivot = grid_df.pivot(index="b", columns="a", values="price")
    grand_mean = pivot.values.mean()
    row_effect = pivot.mean(axis=1)
    col_effect = pivot.mean(axis=0)
    additive = pd.DataFrame(
        np.add.outer(row_effect.values, col_effect.values) - grand_mean,
        index=pivot.index, columns=pivot.columns,
    )
    residual_std = (pivot - additive).values.std()
    total_std = pivot.values.std()
    ratio = residual_std / total_std if total_std > 0 else 0.0
    return ratio


WHATIF_BOOL_KEYS = {"has_license", "host_is_superhost", "host_has_profile_pic", "host_identity_verified", "is_new_listing"}

REVIEW_FIELD_KEYS = set(REVIEW_DEFAULTS.keys())

ROOM_TYPE_PLAIN_LABELS = [v.split(": ", 1)[1] for v in ROOM_TYPE_ICONS.values()]
PROPERTY_TYPE_PLAIN_LABELS = ["Otro (agrupa el resto)" if pt == "Other" else pt for pt in PROPERTY_TYPES]

# Misma redacción que el help= de "Tipo de alojamiento" en el formulario (app.py, sección
# "Alojamiento"): una única fuente de verdad para no describir cada tipo de forma distinta
# en dos sitios de la app.
ROOM_TYPE_DESCRIPTIONS = {
    "Vivienda entera": "el huésped tiene todo el piso para él",
    "Habitación privada": "habitación propia dentro de una vivienda compartida",
    "Habitación compartida": "comparte también la habitación con otros huéspedes",
    "Habitación de hotel": "gestionado como un hotel u hostal",
}

PROPERTY_TYPE_DESCRIPTIONS = {
    "Entire rental unit": "piso de alquiler estándar, vivienda entera",
    "Private room in rental unit": "habitación privada dentro de un piso de alquiler estándar",
    "Entire serviced apartment": "apartamento entero con servicios tipo hotel",
    "Entire condo": "piso entero en un edificio con servicios comunitarios",
    "Room in hotel": "habitación gestionada como en un hotel",
    "Private room in hostel": "habitación privada dentro de un hostel",
    "Entire loft": "loft entero, espacio abierto tipo estudio",
    "Private room in home": "habitación privada dentro de una vivienda particular",
    "Room in boutique hotel": "habitación en un hotel pequeño y con encanto",
    "Entire home": "casa entera para el huésped",
    "Private room in condo": "habitación privada dentro de un piso con servicios comunitarios",
    "Shared room in hostel": "cama u habitación compartida en un hostel",
    "Private room in bed and breakfast": "habitación privada en un alojamiento con desayuno incluido",
    "Otro (agrupa el resto)": "tipos de propiedad poco frecuentes en los datos, agrupados",
}


def bullet_list(values, descriptions=None):
    if descriptions:
        return "\n".join(f"- **{v}**: {descriptions[v]}" for v in values)
    return "\n".join(f"- **{v}**" for v in values)


FEATURE_CATALOG = [
    ("Ubicación", "location_on", [
        (
            "Distrito",
            "Distrito de Barcelona donde está el alojamiento. Una de las variables con más peso en el precio.\n\n"
            "Valores posibles:\n\n" + bullet_list(DISTRICTS),
            "10 distritos",
        ),
        (
            "Barrio",
            "Barrio concreto dentro del distrito. Se usa para calcular el nivel de precios de la zona.\n\n"
            "Valores posibles:\n\n" + bullet_list(sorted(NEIGHBOURHOOD_STATS.keys())),
            "69 barrios",
        ),
        ("Nivel de precios", "**Nivel de precios del barrio**\n\nPrecio medio histórico del barrio. No se introduce a mano: se calcula automáticamente a partir del barrio elegido.", "≈ 80€ – 450€"),
        ("Dist. al centro", "**Distancia al centro**\n\nDistancia en línea recta hasta Plaça Catalunya. Se calcula automáticamente a partir de las coordenadas.", "0 – 8 km aprox."),
    ]),
    ("Alojamiento", "apartment", [
        (
            "Tipo alojamiento",
            "**Tipo de alojamiento**\n\nVivienda entera, habitación privada, compartida o de hotel.\n\n"
            "Valores posibles:\n\n" + bullet_list(ROOM_TYPE_PLAIN_LABELS, ROOM_TYPE_DESCRIPTIONS),
            "4 tipos",
        ),
        (
            "Tipo propiedad",
            "**Tipo de propiedad**\n\nTipo de inmueble concreto (piso, loft, apartamento con servicios, habitación de hostal...).\n\n"
            "Valores posibles:\n\n" + bullet_list(PROPERTY_TYPE_PLAIN_LABELS, PROPERTY_TYPE_DESCRIPTIONS),
            "14 tipos",
        ),
        ("Huéspedes", "Número máximo de personas que pueden alojarse.", "1 – 16"),
        ("Dormitorios", "Número de dormitorios.", "0 – 20"),
        ("Camas", "Número de camas (puede diferir de los dormitorios).", "1 – 20"),
        ("Baños", "Número de baños, admite medios baños.", "0 – 10"),
        ("Noches mín.", "**Estancia mínima**\n\nNoches mínimas por reserva. La variable con más peso de todo el modelo.", "1 – 365 noches"),
        ("Noches máx.", "**Estancia máxima**\n\nNoches máximas que se pueden reservar de una vez.", "1 – 1125 noches"),
        ("Disponibilidad", "**Días disponibles al año**\n\nDisponibilidad del anuncio en el calendario.", "0 – 365 días"),
        ("Licencia", "**Licencia turística**\n\nSi el anuncio tiene número de licencia registrado (requisito legal en Barcelona).", "Sí / No"),
    ]),
    ("Sobre ti como anfitrión", "person", [
        ("Superhost", "Distinción de Airbnb por buenas valoraciones, respuesta rápida y pocas cancelaciones.", "Sí / No"),
        ("Foto perfil", "**Foto de perfil**\n\nSi el perfil del anfitrión tiene foto visible.", "Sí / No"),
        ("Id. verificada", "**Identidad verificada**\n\nSi el anfitrión ha verificado su identidad ante Airbnb.", "Sí / No"),
        ("Nº anuncios", "**Anuncios que gestiona**\n\nCuántos anuncios en total gestiona ese anfitrión, incluido este.", "1 – 500"),
        ("Años anfitrión", "**Años como anfitrión**\n\nTiempo publicando anuncios en Airbnb.", "0 – 20 años"),
        ("Años con cuenta", "**Años con cuenta en la plataforma**\n\nTiempo registrado en Airbnb, aunque no publicara desde el principio.", "0 – 25 años"),
        ("% viv. entera", "**% de sus anuncios que son vivienda entera**\n\nSe calcula automáticamente a partir de cuántos anuncios gestiona ese anfitrión y de qué tipo son.", "0 – 1"),
    ]),
    ("Historial del anuncio", "history", [
        ("Anuncio nuevo", "Si el anuncio todavía no tiene reseñas ni historial de reservas.", "Sí / No"),
        ("Tiene reseñas", "Se deriva automáticamente de si el anuncio es nuevo o no.", "Sí / No"),
        ("Nº reseñas", "**Número de reseñas**\n\nTotal de reseñas recibidas desde la publicación.", "0 – 250+"),
        ("Reseñas/año", "**Reseñas último año**\n\nReseñas recibidas en los últimos 12 meses.", "0 – 50+"),
        ("Reseñas/mes", "**Reseñas al mes**\n\nMedia de reseñas recibidas por mes.", "0 – 8+"),
        ("Valoración", "**Valoración media**\n\nPuntuación media de las reseñas.", "0 – 5"),
        ("Antigüedad", "**Antigüedad del anuncio**\n\nDías desde que se publicó el anuncio.", "0 – 2900+ días"),
        ("Últ. reseña", "**Días desde la última reseña**\n\nCuánto hace de la reseña más reciente.", "0 – 365+ días"),
        ("Ocupación/año", "**Noches ocupadas último año**\n\nEstimación de cuántas noches se ha reservado en el último año.", "0 – 365 noches"),
    ]),
]


def whatif_apply(base_inputs, key, value):
    inp = dict(base_inputs)
    if key == "district":
        candidates = NEIGHBOURHOODS_BY_DISTRICT[value]
        stats_by_n = {n: NEIGHBOURHOOD_STATS[n] for n in candidates}
        mean_price = np.mean([s["price_encoded"] for s in stats_by_n.values()])
        best_n = min(candidates, key=lambda n: abs(stats_by_n[n]["price_encoded"] - mean_price))
        inp["district"] = value
        inp["neighbourhood"] = best_n
        inp["latitude"] = stats_by_n[best_n]["lat"]
        inp["longitude"] = stats_by_n[best_n]["lon"]
    elif key == "neighbourhood":
        stats = NEIGHBOURHOOD_STATS[value]
        inp["neighbourhood"] = value
        inp["district"] = stats["district"]
        inp["latitude"] = stats["lat"]
        inp["longitude"] = stats["lon"]
    elif key in REVIEW_FIELD_KEYS:
        inp["is_new_listing"] = False
        for rk, rv in REVIEW_DEFAULTS.items():
            inp.setdefault(rk, rv)
        inp[key] = value
    elif key == "is_new_listing":
        inp["is_new_listing"] = value
        if not value:
            for rk, rv in REVIEW_DEFAULTS.items():
                inp.setdefault(rk, rv)
    else:
        inp[key] = value
    return inp


def whatif_predict(model, base_inputs, key, value):
    inp = whatif_apply(base_inputs, key, value)
    row = build_features(inp)
    Xrow = pd.DataFrame([row])[FEATURE_COLS]
    for c in Xrow.select_dtypes(include="bool").columns:
        Xrow[c] = Xrow[c].astype(int)
    return float(np.expm1(model.predict(Xrow)[0]))


PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Precio Airbnb Barcelona", layout="wide")


def inject_global_css():
    """Todo el sistema visual de la app en un único sitio: paleta, tarjetas, cabeceras
    de sección y escala del rojo. Se llama una sola vez, justo después de
    set_page_config, para que el estilo viva en un solo lugar."""
    st.markdown(
        """
        <style>
        /* ==================== Base y paleta ==================== */
        .stApp {background-color: #EDEAE4;}
        [data-testid="stHeader"] {background: transparent;}
        .block-container {padding-top: 1.6rem; padding-bottom: 1rem;}
        h2 {margin-top: 0.1rem; margin-bottom: 0.5rem;}
        h3 {margin-top: 0.1rem; margin-bottom: 0.4rem;}
        h4 {margin-top: 0.1rem; margin-bottom: 0.4rem;}
        [data-testid="stMetricValue"] {font-size: 1.9rem;}
        div[data-testid="stHorizontalBlock"] {align-items: stretch;}
        [data-testid="stWidgetLabel"] {
            display: flex !important;
            align-items: center !important;
            width: fit-content !important;
        }
        [data-testid="stWidgetLabel"] > div {
            flex: 0 0 auto !important;
            justify-content: flex-start !important;
            margin-left: 0.3rem !important;
        }

        /* ==================== Barra lateral ==================== */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 0.5px solid #DAD7CE;
        }
        [data-testid="stNavSectionHeader"] {
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-size: 0.72rem !important;
            color: #8A8778 !important;
        }
        [data-testid="stSidebarNavLink"] {
            gap: 0.85rem !important;
            padding-top: 0.6rem !important;
            padding-bottom: 0.6rem !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {
            font-size: 1.5rem !important;
            width: 1.5rem !important;
            height: 1.5rem !important;
        }
        [data-testid="stSidebarNavLink"] span:not([data-testid="stIconMaterial"]) {
            font-size: 1.05rem !important;
        }
        /* Pestaña activa: rosa muy claro + texto rojo oscuro, sin barra lateral roja
        (el rojo pleno se reserva para la acción principal). */
        [data-testid="stSidebarNavLink"][aria-current="page"] {
            background-color: #FCE9ED !important;
        }
        [data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"],
        [data-testid="stSidebarNavLink"][aria-current="page"] span:not([data-testid="stIconMaterial"]) {
            color: #B0203C !important;
        }

        /* ==================== Tarjetas ==================== */
        /* Blancas sobre el fondo cálido, sin borde gris: la separación la da la sombra. */
        [class*="st-key-section_"] {
            background-color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        [class*="st-key-section_form_"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
        [class*="st-key-section_form_"] {
            padding: 0.6rem 1rem !important;
        }
        [class*="st-key-section_catalog_"] {
            padding: 0.5rem 0.75rem !important;
            margin-bottom: 0.4rem !important;
        }

        /* ==================== Cabeceras de sección ==================== */
        /* La cabecera va sobre la propia tarjeta blanca, separada del cuerpo por una
        línea inferior, y llega de borde a borde. Para estirarla hay que aplicar los
        márgenes negativos al .stMarkdown interior, no al .stElementContainer: a ese
        Streamlit le fija el ancho en píxeles vía JS y reafirma el valor, así que
        cualquier ajuste de ancho ahí se pierde. El .stMarkdown es un div normal en
        flujo, así que con width:auto + margen negativo se expande sin pelea.
        Cada variante de tarjeta tiene su propio padding (15px por defecto, 9.6/16px en
        Formulario, 8/12px en el catálogo de Variables): se guardan en variables CSS
        para calcular el margen negativo exacto en vez de fijar valores a ciegas.
        Ojo con la especificidad: :not() suma, así que las variantes tienen que repetirlo
        o perderían contra la regla general aun yendo después. */
        [class*="st-key-section_"]:not([class*="st-key-section_home_"]) {
            --card-pad-y: 15px;
            --card-pad-x: 15px;
        }
        [class*="st-key-section_form_"]:not([class*="st-key-section_home_"]) {
            --card-pad-y: 9.6px;
            --card-pad-x: 16px;
        }
        [class*="st-key-section_catalog_"]:not([class*="st-key-section_home_"]) {
            --card-pad-y: 8px;
            --card-pad-x: 12px;
        }
        [class*="st-key-section_"]:not([class*="st-key-section_home_"]) > .stElementContainer:first-child .stMarkdown {
            margin: calc(-1 * var(--card-pad-y)) calc(-1 * var(--card-pad-x)) 0 calc(-1 * var(--card-pad-x));
            padding: 0.55rem var(--card-pad-x) 0.5rem var(--card-pad-x);
            border-bottom: 1px solid #E4E1D9;
            border-radius: 12px 12px 0 0;
        }
        [class*="st-key-section_"]:not([class*="st-key-section_home_"]) > .stElementContainer:first-child .stMarkdown :is(h1, h2, h3, h4, h5, p) {
            margin: 0 !important;
            color: #2C2C2A;
        }

        /* ==================== Tarjeta héroe: el precio ==================== */
        /* Única cabecera con fondo propio: gris oscuro, para que el precio destaque
        sobre el resto de secciones. El número va en negro, en el cuerpo blanco. */
        .st-key-section_prediccion_resumen > .stElementContainer:first-child .stMarkdown {
            background-color: #3A3A37 !important;
            border-bottom: none !important;
            /* Franja más fina que el resto de cabeceras: sigue destacando por el color,
            pero ocupa menos alto para dejar sitio a las secciones de abajo. */
            padding-top: 0.28rem !important;
            padding-bottom: 0.25rem !important;
        }
        .st-key-section_prediccion_resumen > .stElementContainer:first-child .stMarkdown :is(h4, p) {
            color: #F1EFE8 !important;
        }
        .st-key-section_prediccion_resumen h1 {
            color: #2C2C2A !important;
            font-size: 3rem !important;
            line-height: 1.1 !important;
            padding: 0.2rem 0 0 0 !important;
            margin: 0 !important;
        }

        /* ==================== Cajas interiores ==================== */
        /* Chips de variables y tarjetas de métricas: gris cálido propio, ni blanco (se
        fundiría con la tarjeta) ni el gris del fondo de página. */
        [data-testid="stMetric"] {
            background-color: #F1EFE8 !important;
            border: none !important;
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
        }
        .st-key-variable_catalog [data-testid="stMetric"] {
            padding: 0.3rem 0.5rem !important;
            gap: 0.05rem;
        }
        .st-key-variable_catalog [data-testid="stMetricLabel"] {font-size: 0.75rem;}
        .st-key-variable_catalog [data-testid="stMetricValue"] {font-size: 0.95rem; line-height: 1.15;}

        /* ==================== Escala del rojo ==================== */
        /* Toggles: rojo claro al estar activos, para no competir con el botón principal
        (el único con rojo pleno). El track es el div directo dentro de
        <label data-baseweb="checkbox"> que a su vez contiene otro div (el thumb): esa
        forma distingue un toggle de un st.checkbox normal, cuyo primer hijo es un
        <span> (icono de casilla). */
        label[data-baseweb="checkbox"]:has(> input:checked) > div:first-child:has(> div) {
            background-color: #F7AEBC !important;
            border-color: #F7AEBC !important;
        }
        label[data-baseweb="checkbox"]:has(> input:not(:checked)) > div:first-child:has(> div) {
            background-color: #D3D1C7 !important;
            border-color: #D3D1C7 !important;
        }
        /* Checkbox marcado: mismo criterio que el toggle (su primer hijo es un <span>,
        no un <div>, que es justo lo que los distingue). */
        label[data-baseweb="checkbox"]:has(> input:checked) > span:first-child {
            background-color: #F7AEBC !important;
            border-color: #F7AEBC !important;
        }
        /* Pill seleccionada: mismo tratamiento que la pestaña activa de la nav. */
        [data-testid="stBaseButton-pillsActive"] {
            background-color: #FCE9ED !important;
            border-color: #B0203C !important;
            color: #B0203C !important;
        }

        /* ==================== Tarjetas de Inicio (enlaces) ==================== */
        [class*="st-key-section_home_"] {
            position: relative;
            transition: box-shadow 0.15s ease;
        }
        [class*="st-key-section_home_"]:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10);
        }
        [class*="st-key-section_home_"] .stElementContainer:has(a[data-testid="stPageLink-NavLink"]) {
            /* Streamlit pone position:relative aquí; si no se anula, ese contenedor (que se
            colapsa a 0 al sacar el enlace del flujo) pasa a ser el bloque contenedor del
            enlace absoluto, en vez de la tarjeta entera. */
            position: static !important;
            /* El enlace absoluto ya no aporta altura al flujo normal: se reserva a mano el
            hueco del título para que el texto de abajo no quede pegado/solapado. */
            min-height: 2rem;
        }
        [class*="st-key-section_home_"] a[data-testid="stPageLink-NavLink"] {
            /* inset con el mismo valor que el padding de la tarjeta (15px, el de
            st.container(border=True)): un enlace absoluto se posiciona respecto al borde
            de la caja de padding, así que con inset:0 el título quedaba pegado al borde,
            ignorando ese padding. Con width/height en auto (no 100%), se recalculan solos
            a partir de estos 4 huecos, dejando la misma separación que el resto del texto. */
            position: absolute !important;
            inset: 15px !important;
            width: auto !important;
            height: auto !important;
            z-index: 1;
            align-items: flex-start !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()


@st.cache_resource
def load_model():
    return joblib.load(PROJECT_DIR / "models" / "price_model.pkl")


@st.cache_data
def load_comparables():
    cols = [
        "id", "price", "room_type", "property_type", "neighbourhood_cleansed",
        "neighbourhood_group_cleansed", "accommodates", "bedrooms", "bathrooms",
        "minimum_nights", "latitude", "longitude", "name", "picture_url", "listing_url",
        "estimated_occupancy_l365d", "review_scores_rating", "has_license",
    ]
    df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "listings_full_clean.csv", usecols=cols)
    df["price_weight"] = np.log1p(df["price"])
    return df


@st.cache_data
def load_district_geojson():
    with open(APP_DIR / "assets" / "districts.geojson", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_neighbourhood_geojson():
    with open(APP_DIR / "assets" / "neighbourhoods_colored.geojson", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    with open(APP_DIR / "assets" / "feature_importance.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_interaction_scores():
    with open(APP_DIR / "assets" / "interaction_scores.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test_matrix():
    test_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "listings_test.csv")
    X = test_df[FEATURE_COLS].copy()
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    return X


@st.cache_data
def load_test_evaluation():
    """Sesgo del modelo (predicho - real) en el set de test, por distrito y por
    quintil de precio. Mismo análisis que 03_model_evaluation.ipynb, recalculado
    aquí para poder mostrarlo en la app."""
    test_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "listings_test.csv")
    X = test_df[FEATURE_COLS].copy()
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    pred = np.expm1(model.predict(X))
    actual = test_df["price"].values
    residual = pred - actual

    district_cols = [c for c in FEATURE_COLS if c.startswith("district_")]
    district = X[district_cols].idxmax(axis=1).str.replace("district_", "", regex=False)

    quintile = pd.qcut(actual, 5, labels=["Q1 (más baratos)", "Q2", "Q3", "Q4", "Q5 (más caros)"])

    return pd.DataFrame({
        "district": district.values,
        "quintile": quintile.astype(str),
        "residual": residual,
        "price": actual,
    })


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["model"])


@st.cache_data
def load_shap_summary(sample_size=400):
    """Contribución SHAP de cada variable numérica/booleana en una muestra del test set,
    con el valor real de esa variable normalizado (0-1) para colorear cada punto."""
    X_test = load_test_matrix()
    n = min(sample_size, len(X_test))
    X_sample = X_test.sample(n=n, random_state=42).reset_index(drop=True)
    explainer = get_shap_explainer(model)
    shap_values = explainer(X_sample).values
    col_index = {c: i for i, c in enumerate(X_sample.columns)}

    rng = np.random.default_rng(42)
    frames = []
    for label, spec in SHAP_SUMMARY_GROUPS.items():
        if spec["kind"] == "numeric":
            cols = spec["cols"]
            value_col = spec["sample_key"]
        else:
            cols = [spec["col"]]
            value_col = spec["col"]
        contrib = sum(shap_values[:, col_index[c]] for c in cols)
        raw_values = X_sample[value_col].to_numpy(dtype=float)
        vmin, vmax = raw_values.min(), raw_values.max()
        value_norm = (raw_values - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_values)
        frames.append(pd.DataFrame({
            "feature": label,
            "shap": contrib,
            "value_norm": value_norm,
            "jitter": rng.uniform(-0.4, 0.4, size=n),
        }))

    df = pd.concat(frames, ignore_index=True)
    order = (
        df.groupby("feature")["shap"].apply(lambda s: s.abs().mean()).sort_values(ascending=False).index.tolist()
    )
    return df, order


BLUE = "#3b6ea5"
RED = "#c0392b"
GREEN = "#1a7f37"
# What-if: dos intensidades de azul en vez de rojo, para dejar el rojo libre como
# "baja el precio" en el waterfall de al lado y que no se confundan.
WHATIF_CURRENT = "#185FA5"  # "tu valor actual" (gráfico de línea, numéricas)
WHATIF_ALT = "#B5D4F4"  # las alternativas (ambos gráficos what-if)
WHATIF_SIM = "#BA7517"  # valor simulado/activo en el gráfico de barras (categóricas):
# naranja en vez de azul para no confundirlo con "tu valor actual" del gráfico de línea
MARKET_COLOR = "#8e44ad"  # codificación de datos para "mercado de temporada / larga estancia", ver page_modelo
GRAY_MUTED = "#9aa0a6"  # zona de transición 8-30 noches: poco frecuente, casi sin datos
WF_START = "#C4C0B6"  # waterfall: punto de partida (gris neutro, no es un cambio de precio)
WF_TOTAL = "#3A3A37"  # waterfall: precio final (gris oscuro; evita rojo, que ahí significa "baja")

MARKET_CUTOFF = 30  # noches: <=30 turístico, >=31 temporada (frontera legal VUT en Barcelona)

BEDROOMS_SLIDER_MAX = 6  # filtro "Tamaño" del Mapa del mercado: el tope del slider es "6+"
MIN_LISTINGS_PER_BARRIO = 5  # por debajo de esto, un barrio se pinta "sin datos": la media
# ya no es fiable calculada sobre 2-3 anuncios tras aplicar los filtros


def market_label(minimum_nights):
    if minimum_nights <= MARKET_CUTOFF:
        return "🏖️ Mercado turístico (corta estancia)"
    return "🏠 Mercado de temporada (larga estancia)"


@st.cache_data
def neighbourhood_market_agg(room_type, accommodates_range, bedrooms_range):
    """Agregados por barrio para el Mapa del mercado. room_type='Todos' o uno de
    ROOM_TYPES, accommodates_range=(min, max) de huéspedes, bedrooms_range=(min, max)
    con max=BEDROOMS_SLIDER_MAX significando "sin tope superior" (el "6+" del slider):
    recalcula todos los agregados solo con los anuncios que cumplen los tres filtros,
    sin tocar comparables_all. avg_rating usa mean() con skipna, ignorando anuncios
    nuevos sin reseñas todavía en vez de arrastrar NaN al agregado del barrio."""
    df = comparables_all
    if room_type != "Todos":
        df = df[df["room_type"] == room_type]
    acc_lo, acc_hi = accommodates_range
    df = df[df["accommodates"].between(acc_lo, acc_hi)]
    bed_lo, bed_hi = bedrooms_range
    if bed_hi >= BEDROOMS_SLIDER_MAX:
        df = df[df["bedrooms"] >= bed_lo]
    else:
        df = df[df["bedrooms"].between(bed_lo, bed_hi)]
    return (
        df.groupby("neighbourhood_cleansed")
        .agg(
            avg_price=("price", "mean"),
            n_listings=("price", "size"),
            pct_corta_estancia=("minimum_nights", lambda s: (s <= MARKET_CUTOFF).mean()),
            pct_entire_home=("room_type", lambda s: (s == "Entire home/apt").mean()),
            avg_occupancy=("estimated_occupancy_l365d", "mean"),
            avg_rating=("review_scores_rating", "mean"),
            pct_license=("has_license", "mean"),
        )
        .rename_axis("neighbourhood")
        .reset_index()
    )


def find_similar_comparables(room_types, acc_range, mn_range, radius):
    """Anuncios reales parecidos al anuncio confirmado del usuario (mismo tipo,
    tamaño y estancia mínima similares), para comparar su precio contra el
    mercado. radius: "Tu barrio" / "Tu distrito" / "Toda la ciudad". Compartida
    por la verificación de Predicción y por la página Comparar, para no repetir
    la misma lógica de filtrado en dos sitios."""
    if radius == "Tu barrio":
        geo_mask = comparables_all["neighbourhood_cleansed"] == neighbourhood
    elif radius == "Toda la ciudad":
        geo_mask = pd.Series(True, index=comparables_all.index)
    else:  # "Tu distrito"
        geo_mask = comparables_all["neighbourhood_group_cleansed"] == district
    acc_lo, acc_hi = acc_range
    mn_lo, mn_hi = mn_range
    mask = (
        geo_mask
        & comparables_all["room_type"].isin(room_types)
        & comparables_all["accommodates"].between(acc_lo, acc_hi)
        & comparables_all["minimum_nights"].between(mn_lo, mn_hi)
    )
    return comparables_all[mask]


def chart_market_bins(df):
    """Barras de nº de anuncios por tramo de minimum_nights, coloreadas por grupo de
    mercado (turístico / zona de transición casi sin datos / temporada), para enseñar
    de un vistazo el valle entre los dos mercados que conviven en el dataset."""
    order = df["tramo"].tolist()
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("n:Q", title="Nº de anuncios"),
            y=alt.Y("tramo:N", sort=order, title=None),
            color=alt.Color(
                "grupo:N",
                scale=alt.Scale(
                    domain=["Turístico", "Zona de transición", "Temporada"],
                    range=[BLUE, GRAY_MUTED, MARKET_COLOR],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("tramo:N", title=""),
                alt.Tooltip("n:Q", title="Nº de anuncios"),
                alt.Tooltip("mediana_precio:Q", title="Mediana €/noche", format=".0f"),
            ],
        )
        .properties(height=alt.Step(28))
    )


def chart_hbar(labels, values, x_title, color=BLUE):
    df = pd.DataFrame({"label": labels, "value": values})
    return (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusEnd=3)
        .encode(
            x=alt.X("value:Q", title=x_title),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", title=x_title, format=".2f")],
        )
        .properties(height=alt.Step(24))
    )


def chart_hbar_highlight(labels, values, sim_label, x_title):
    """sim_label es el valor SIMULADO (el del desplegable "Simula un valor"), no el
    guardado en el formulario: ese se muestra aparte, como texto, porque puede quedar
    fuera de la ventana visible de barras."""
    df = pd.DataFrame({"label": labels, "value": values})
    df["estado"] = df["label"].apply(lambda l: "Valor simulado" if l == sim_label else "Alternativa")
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("value:Q", title=x_title),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250)),
            color=alt.Color(
                "estado:N",
                scale=alt.Scale(domain=["Valor simulado", "Alternativa"], range=[WHATIF_SIM, WHATIF_ALT]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", title=x_title, format=".0f")],
        )
        .properties(height=alt.Step(20))
    )


def chart_bias_hbar(labels, values, x_title, sort="x"):
    df = pd.DataFrame({"label": labels, "value": values})
    df["sesgo"] = df["value"].apply(lambda v: "Sobreestima" if v >= 0 else "Subestima")
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("value:Q", title=x_title),
            y=alt.Y("label:N", sort=sort, title=None, axis=alt.Axis(labelLimit=200)),
            color=alt.Color(
                "sesgo:N",
                scale=alt.Scale(domain=["Sobreestima", "Subestima"], range=[BLUE, RED]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", title=x_title, format=".1f")],
        )
        .properties(height=alt.Step(24))
    )


def chart_waterfall(df, x_title):
    """Tres significados de color: gris = punto de partida, verde/rojo = el paso sube o
    baja el precio, gris oscuro = total. Las etiquetas de valor van siempre por fuera del
    extremo derecho de cada barra, así nunca se solapan aunque la barra sea muy corta."""
    order = df["label"].tolist()
    color = alt.Color(
        "kind:N",
        scale=alt.Scale(
            domain=["punto de partida", "sube", "baja", "total"],
            range=[WF_START, GREEN, RED, WF_TOTAL],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=250))
    # Margen a la derecha para que quepan las etiquetas fuera de la barra más larga; el
    # contorno "antes" (what-if) puede llegar más lejos que cualquier "end", así que
    # también cuenta para el máximo de la escala.
    x_max = float(df["end"].max())
    if "before_end" in df.columns and df["before_end"].notna().any():
        x_max = max(x_max, float(df["before_end"].max()))
    x_scale = alt.Scale(domain=[0, x_max * 1.15], nice=False)
    x = alt.X("start:Q", title=x_title, scale=x_scale)

    pasos = df[df["kind"].isin(["sube", "baja"])]
    extremos = df[df["kind"].isin(["punto de partida", "total"])]

    # Dos capas de barras solo por el tooltip: los pasos tienen salto acumulado que
    # enseñar y los extremos no, y en Vega el tooltip se define por capa.
    # title=" " en vez de "": con título vacío Vega cae de vuelta al nombre del campo
    # ("tip1") y lo pinta como etiqueta de la fila del tooltip.
    # Vega-Lite/vega-tooltip escapan cualquier HTML puesto en el valor de un tooltip
    # (comprobado: "<b>...</b>" sale literal, como texto, no en negrita) y
    # st.altair_chart no expone la opción sanitize:false para desactivarlo — así que
    # aquí no hay negrita ni color, solo texto plano en 3 líneas.
    bars_pasos = alt.Chart(pasos).mark_bar(cornerRadiusEnd=3).encode(
        x=x, x2="end:Q", y=y, color=color,
        tooltip=[
            alt.Tooltip("tip1:N", title=" "),
            alt.Tooltip("tip2:N", title="  "),
            alt.Tooltip("tip3:N", title="   "),
        ],
    )
    # Los extremos no tienen salto que enseñar: con un único campo (no una lista) Vega
    # pinta solo el valor, sin tabla clave/valor.
    bars_extremos = alt.Chart(extremos).mark_bar(cornerRadiusEnd=3).encode(
        x=x, x2="end:Q", y=y, color=color,
        tooltip=alt.Tooltip("tip1:N"),
    )
    # Las etiquetas llevan color propio (scale=None pasa el hex tal cual y no añade una
    # segunda leyenda): el gris de la barra de partida sería ilegible como texto.
    etiquetas = alt.Chart(df).mark_text(align="left", baseline="middle", dx=5, fontSize=11).encode(
        x=alt.X("end:Q", title=x_title, scale=x_scale),
        y=y, text="etiqueta:N",
        color=alt.Color("color_etiqueta:N", scale=None, legend=None),
    )
    layers = [bars_extremos, bars_pasos]

    # What-if: contorno punteado sin relleno que marca dónde llegaba la fila (variable
    # simulada + total) ANTES del cambio, ancladas en el mismo "start" que la barra
    # sólida nueva para poder comparar longitudes desde el mismo punto de partida. No
    # lleva la codificación de color por "kind": es siempre gris, sin entrada en la
    # leyenda, así que no compite visualmente con sube/baja/total.
    if "before_end" in df.columns:
        antes_df = df[df["before_end"].notna()]
        if len(antes_df):
            # Usa su propia columna "before_start" en vez de "start": el "start" de la
            # barra sólida está normalizado con min/max para dibujarse bien en cualquier
            # dirección y no siempre coincide con el punto de partida real de este paso.
            antes = alt.Chart(antes_df).mark_bar(
                filled=False, stroke="#9aa0a6", strokeDash=[4, 3], strokeWidth=1.5, cornerRadiusEnd=3,
            ).encode(
                x=alt.X("before_start:Q", scale=x_scale), x2="before_end:Q", y=y,
                tooltip=alt.Tooltip("before_tip:N", title=" "),
            )
            layers.append(antes)

    layers.append(etiquetas)
    return alt.layer(*layers).properties(height=alt.Step(20))


def chart_shap_summary(df, order):
    return (
        alt.Chart(df)
        .mark_circle(size=22, opacity=0.55)
        .encode(
            x=alt.X("shap:Q", title="Impacto en el precio predicho (€)"),
            y=alt.Y(
                "feature:N", sort=order, title=None,
                axis=alt.Axis(labelExpr="datum.label == 'Estancia mínima' ? datum.label + ' *' : datum.label"),
            ),
            yOffset=alt.YOffset("jitter:Q"),
            color=alt.Color(
                "value_norm:Q",
                title="Valor de la variable",
                scale=alt.Scale(domain=[0, 1], range=[BLUE, RED]),
                legend=alt.Legend(orient="top", gradientLength=140, labelExpr="datum.value == 0 ? 'Bajo' : datum.value == 1 ? 'Alto' : ''"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Variable"),
                alt.Tooltip("shap:Q", title="Impacto (€)", format=".1f"),
            ],
        )
        .properties(height=alt.Step(26))
    )


def chart_line_highlight(x_values, y_values, current_x, x_title, y_title="Precio predicho (€)"):
    df = pd.DataFrame({"x": x_values, "y": y_values})
    # La curva entera son las alternativas (azul claro); solo el punto del valor
    # actual va en azul fuerte, igual que en el gráfico de barras del what-if.
    line = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(color=WHATIF_ALT, size=50), color=WHATIF_ALT)
        .encode(
            x=alt.X("x:Q", title=x_title),
            y=alt.Y("y:Q", title=y_title),
            tooltip=[alt.Tooltip("x:Q", title=x_title), alt.Tooltip("y:Q", title=y_title, format=".0f")],
        )
    )
    current_df = df[df["x"] == current_x]
    point = (
        alt.Chart(current_df)
        .mark_point(size=140, color=WHATIF_CURRENT, filled=True)
        .encode(x="x:Q", y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Tu valor actual", format=".0f")])
    )
    return (line + point).properties(height=220)


model = load_model()
comparables_all = load_comparables()

DEFAULT_INPUTS = {
    "district": "Eixample",
    "neighbourhood": NEIGHBOURHOODS_BY_DISTRICT["Eixample"][0],
    "room_type": ROOM_TYPES[0],
    "property_type": PROPERTY_TYPES[0],
    "accommodates": 4,
    "bedrooms": 1,
    "beds": 2,
    "bathrooms": 1.0,
    "minimum_nights": 2,
    "maximum_nights": 365,
    "availability_365": 300,
    "has_license": True,
    "host_is_superhost": False,
    "host_has_profile_pic": True,
    "host_identity_verified": True,
    "calculated_host_listings_count": 1,
    "host_tenure_years": 0.0,
    "host_user_tenure_years": 0.5,
    "host_entire_homes_ratio": 1.0,
    "is_new_listing": True,
    "latitude": NEIGHBOURHOOD_STATS[NEIGHBOURHOODS_BY_DISTRICT["Eixample"][0]]["lat"],
    "longitude": NEIGHBOURHOOD_STATS[NEIGHBOURHOODS_BY_DISTRICT["Eixample"][0]]["lon"],
}

def page_home():
    st.title("Recomendador de precio · Airbnb Barcelona")
    st.markdown(
        "Esta herramienta te ayuda a decidir a qué precio publicar (o repreciar) tu anuncio en Barcelona, "
        "comparándolo con más de 13.000 anuncios reales de la ciudad. Está pensada tanto para un anfitrión "
        "que publica su primer anuncio como para uno que quiere revisar si su precio actual tiene sentido."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True, key="section_home_form"):
            st.page_link(page_formulario_p, label="1 · Formulario", icon=":material/edit_note:")
            st.caption("Introduce las características de tu alojamiento: ubicación, tamaño, condiciones de reserva y tu perfil como anfitrión.")
    with c2:
        with st.container(border=True, key="section_home_pred"):
            st.page_link(page_prediccion_p, label="2 · Predicción", icon=":material/payments:")
            st.caption("Obtén un precio recomendado con un rango orientativo, avisos de confianza y qué variables lo explican.")
    with c3:
        with st.container(border=True, key="section_home_map"):
            st.page_link(page_comparar_p, label="3 · Comparar", icon=":material/compare:")
            st.caption("Comprueba si tu precio encaja con anuncios similares cerca de ti, con mapa incluido.")

    st.divider()
    st.page_link(
        page_modelo_p,
        label="¿Quieres saber cómo funciona y si te puedes fiar? → Cómo funciona el modelo",
        icon=":material/account_tree:",
    )


def page_modelo():
    st.header("Cómo funciona el modelo")
    st.caption(
        "Antes de fiarte de un precio, es normal preguntarse qué hay detrás. Aquí tienes, sin tecnicismos, "
        "qué es este modelo, cuánto se equivoca, dónde falla más y cómo se construyó."
    )

    with st.container(border=True, key="section_modelo_que_es"):
        st.markdown("#### :material/account_tree: Qué es el modelo")
        st.markdown(
            "El precio que ves no sale de una fórmula fija ni de una media simple: lo calcula un modelo "
            "entrenado con más de 13.000 anuncios reales de Airbnb en Barcelona, que ha aprendido qué "
            "combinaciones de ubicación, tamaño, condiciones de reserva e historial del anfitrión se han "
            "pagado realmente en la ciudad."
        )
        st.caption(f"En detalle técnico, es un {METADATA['model']} (una familia de modelos de árboles de decisión).")

    with st.container(border=True, key="section_modelo_acierto"):
        st.markdown("#### :material/track_changes: Cuánto acierta")
        mae = METADATA["test_metrics"]["MAE"]
        r2 = METADATA["test_metrics"]["R2"]
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Error medio por noche", f"{mae:.0f} €")
        mcol2.metric("Diferencias de precio que explica", f"{r2*100:.0f}%")
        st.caption(
            f"En anuncios que el modelo nunca vio durante el entrenamiento, se equivoca de media unos "
            f"{mae:.0f}€ por noche (arriba o abajo), y explica un {r2*100:.0f}% de las diferencias de precio "
            "que hay entre anuncios distintos en Barcelona."
        )

    with st.container(border=True, key="section_modelo_fiabilidad"):
        st.markdown("#### :material/balance: Dónde es más y menos fiable")
        st.markdown(
            "Ningún modelo acierta siempre, y prefiere decírtelo claro a llevarte una sorpresa:"
        )
        st.markdown(
            "- **Alojamientos muy grandes** (10 o más huéspedes): el modelo tiende a quedarse corto en este "
            "tipo de anuncios, más escasos en los datos de entrenamiento.\n"
            "- **Distritos con pocos anuncios**, como Horta-Guinardó o Nou Barris: al haber menos ejemplos, "
            "la predicción es menos precisa que en distritos con más oferta."
        )
        st.caption(
            "La app te avisa automáticamente de estos casos en la página de Predicción, con una señal de confianza."
        )
        st.caption(
            "Sesgo medio del modelo en anuncios reales de test que no usó para entrenar (precio predicho menos "
            "precio real). Un valor negativo grande significa que el modelo tiende a quedarse corto en ese grupo, "
            "no que tu anuncio concreto vaya a fallar igual."
        )
        eval_df = load_test_evaluation()
        col_dist, col_quint = st.columns(2)
        with col_dist:
            by_district = eval_df.groupby("district")["residual"].agg(["mean", "count"]).reset_index()
            district_labels = [f"{row.district} ({row.count})" for row in by_district.itertuples()]
            chart = chart_bias_hbar(district_labels, by_district["mean"], x_title="Sesgo medio (€)")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Entre paréntesis, nº de anuncios de test en ese distrito.")
        with col_quint:
            quintile_order = ["Q1 (más baratos)", "Q2", "Q3", "Q4", "Q5 (más caros)"]
            by_quintile = eval_df.groupby("quintile", sort=False).agg(
                sesgo=("residual", "mean"), precio_min=("price", "min"), precio_max=("price", "max")
            ).reindex(quintile_order)
            quintile_labels = [
                f"{q} · {int(row.precio_min)}–{int(row.precio_max)}€" for q, row in by_quintile.iterrows()
            ]
            chart = chart_bias_hbar(
                quintile_labels, by_quintile["sesgo"], x_title="Sesgo medio (€)",
                sort=list(reversed(quintile_labels)),
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Quintiles de precio real: Q1 = 20% más baratos, Q5 = 20% más caros.")

    with st.container(border=True, key="section_modelo_mercados"):
        st.markdown("#### :material/call_split: Dos mercados en un mismo modelo")
        st.markdown(
            "Hay una razón de fondo por la que el modelo se apoya tanto en la estancia mínima:\n\n"
            "Dentro de estos anuncios conviven dos negocios distintos, aunque el dato en bruto no los separe:\n\n"
            "- **Turístico** (estancia mínima de 30 noches o menos): compite con hoteles, precios por noche altos.\n"
            "- **Temporada / larga estancia** (31 noches o más): compite con alquiler residencial normal, precios "
            "por noche mucho más bajos.\n\n"
            "La frontera no es casualidad: en Barcelona, un alquiler de 31 noches o más deja de considerarse "
            "vivienda de uso turístico a efectos legales — y con la retirada de las licencias turísticas prevista "
            "para 2028, esta distinción va a pesar todavía más."
        )

        mn_all = comparables_all["minimum_nights"]
        price_all = comparables_all["price"]
        bins = [0, 7, 30, 90, 10_000]
        tramo_labels = ["1-7 noches", "8-30 noches", "31-90 noches", "90+ noches"]
        grupo_by_tramo = {
            "1-7 noches": "Turístico", "8-30 noches": "Zona de transición",
            "31-90 noches": "Temporada", "90+ noches": "Temporada",
        }
        tramo = pd.cut(mn_all, bins=bins, labels=tramo_labels)
        bins_df = pd.DataFrame({"tramo": tramo, "price": price_all})
        bins_summary = bins_df.groupby("tramo", observed=True).agg(
            n=("price", "size"), mediana_precio=("price", "median")
        ).reindex(tramo_labels).reset_index()
        bins_summary["grupo"] = bins_summary["tramo"].map(grupo_by_tramo)

        st.altair_chart(chart_market_bins(bins_summary), use_container_width=True)
        pct_transicion = bins_summary.loc[bins_summary["tramo"] == "8-30 noches", "n"].iloc[0] / len(comparables_all) * 100
        st.caption(
            f"Casi todos los anuncios se concentran en 1-2 noches o justo en 31-32 noches; la zona intermedia "
            f"(8-30 noches) apenas tiene datos ({pct_transicion:.1f}% del total). Por eso Estancia mínima es la "
            "variable que más pesa en el modelo: en gran parte, indica a qué mercado pertenece el anuncio, no un "
            "ajuste fino de precio."
        )
        st.caption(
            "Se comprobó entrenar un modelo separado para cada mercado y no mejoraba la precisión frente al "
            "modelo único actual: al ser un modelo de árboles, ya distingue el mercado internamente antes de "
            "afinar el precio, así que no compensa mantener dos modelos por separado."
        )


def page_variables():
    st.header("Variables del modelo")
    st.caption(
        "Todo lo que el modelo tiene en cuenta para calcular un precio. Pasa el ratón por encima de cada "
        "variable (icono ?) para ver su significado y valores posibles. El análisis de cómo pesa e interactúa "
        "cada una está en la pestaña **Análisis**."
    )

    with st.container(key="variable_catalog"):
        n_cols = 3
        col_left, col_right = st.columns(2)
        target_col = {0: col_left, 1: col_left, 2: col_right, 3: col_right}
        for idx, (category, icon, variables) in enumerate(FEATURE_CATALOG):
            with target_col[idx]:
                with st.container(border=True, key=f"section_catalog_{idx}"):
                    st.markdown(f"#### :material/{icon}: {category}")
                    rows = [variables[i:i + n_cols] for i in range(0, len(variables), n_cols)]
                    for row in rows:
                        cols = st.columns(n_cols)
                        for col, (name, desc, rng) in zip(cols, row):
                            with col:
                                st.metric(name, rng, help=desc, border=True)


def page_analisis():
    st.header("Análisis del modelo")
    st.caption(
        "Cómo piensa el modelo por dentro: qué variables pesan más, cómo afecta cada una al precio, y si hay "
        "interacción entre ellas. Para saber si puedes fiarte del modelo en general, ve a Modelo; para el "
        "precio de tu anuncio, ve a Predicción."
    )

    col_imp, col_pdp = st.columns(2)
    with col_imp:
        with st.container(border=True, key="section_analisis_importancia"):
            st.markdown("#### :material/bar_chart: Qué variables pesan más")
            importance = load_feature_importance()
            top_n_imp = 12
            st.caption(
                f"Top {top_n_imp} de las 29 variables (agrupando, por ejemplo, las 10 columnas de distrito en una "
                "sola barra), dejando fuera anuncios reales de prueba. Foto general del modelo, no de tu anuncio."
            )
            show_all_imp = st.checkbox(f"Ver las {len(importance)} variables", key="show_all_importance")
            shown_importance = importance if show_all_imp else importance[:top_n_imp]
            chart = chart_hbar(
                [row["feature"] for row in shown_importance],
                [row["importance"] for row in shown_importance],
                x_title="Importancia (caída de R²)",
            )
            st.altair_chart(chart, use_container_width=True)

    with col_pdp:
        with st.container(border=True, key="section_analisis_pdp"):
            st.markdown("#### :material/show_chart: Cómo afecta cada variable")
            st.caption(
                "Precio medio predicho para los anuncios de test si a todos se les asignara el mismo valor en "
                "esta variable. La franja de abajo muestra cuántos anuncios reales respaldan cada tramo: si hay "
                "pocos, ese tramo de la curva es poco fiable (casi extrapolación)."
            )
            pdp_choice = st.selectbox("Variable a explorar", list(PDP_GROUPS.keys()), key="pdp_choice")
            X_test_matrix = load_test_matrix()
            pdp_values, pdp_results, pdp_kind = pdp_curve(model, X_test_matrix, pdp_choice)
            raw_values, _ = pdp_axis_values(pdp_choice)

            if pdp_kind == "numeric":
                line_chart = (
                    alt.Chart(pd.DataFrame({"x": pdp_values, "y": pdp_results}))
                    .mark_line(point=alt.OverlayMarkDef(color=BLUE, size=50), color=BLUE)
                    .encode(
                        x=alt.X("x:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
                        y=alt.Y("y:Q", title="Precio medio predicho (€)"),
                        tooltip=[alt.Tooltip("x:Q", title=pdp_choice), alt.Tooltip("y:Q", title="Precio medio", format=".0f")],
                    )
                    .properties(height=190)
                )
                real_values = pdp_real_values(X_test_matrix, pdp_choice)
                # Variables de conteo (Dormitorios, Huéspedes...) son enteras de verdad: sin esto,
                # el bineado automático de Vega puede etiquetar el eje con decimales ("1.0", "3.0"...)
                # que no existen en la variable. Las genuinamente continuas (Baños...) sí los conservan.
                is_integer_valued = (real_values.dropna() % 1 == 0).all()
                hist_axis = alt.Axis(format="d") if is_integer_valued else alt.Axis()
                hist_chart = (
                    alt.Chart(pd.DataFrame({"x": real_values}))
                    .mark_bar(color=BLUE, opacity=0.45)
                    .encode(
                        x=alt.X("x:Q", title=pdp_choice, bin=alt.Bin(maxbins=25), axis=hist_axis),
                        y=alt.Y("count():Q", title="Nº anuncios"),
                        tooltip=[alt.Tooltip("count():Q", title="Nº anuncios")],
                    )
                    .properties(height=80)
                )
                pdp_chart = alt.vconcat(line_chart, hist_chart, spacing=4).resolve_scale(x="shared")
            else:
                counts = pdp_category_counts(X_test_matrix, pdp_choice, raw_values)
                labels_with_counts = [f"{lbl} ({c})" for lbl, c in zip(pdp_values, counts)]
                pdp_chart = chart_hbar(labels_with_counts, pdp_results, x_title="Precio medio predicho (€)")
            st.altair_chart(pdp_chart, use_container_width=True)
            if pdp_choice == "Estancia mínima":
                st.caption(
                    "El desplome de la curva y los dos picos del histograma reflejan los dos mercados "
                    "(turístico vs temporada). Más detalle en **Modelo**."
                )

    with st.container(border=True, key="section_analisis_shap"):
        st.markdown("#### :material/scatter_plot: Impacto y dirección de cada variable")
        shap_df, shap_order = load_shap_summary()
        top_n_shap = 12
        st.caption(
            f"Top {top_n_shap} de las {len(shap_order)} variables numéricas o Sí/No con más impacto. Cada punto es "
            "un anuncio de test: su posición indica cuánto sube o baja el precio predicho, y el color si el valor "
            "de la variable era bajo (azul) o alto (rojo). Las agrupadas (distrito, tipo...) ya están en el "
            "ranking de importancia de la izquierda."
        )
        show_all_shap = st.checkbox(f"Ver las {len(shap_order)} variables", key="show_all_shap")
        shown_labels = shap_order if show_all_shap else shap_order[:top_n_shap]
        shap_chart = chart_shap_summary(shap_df[shap_df["feature"].isin(shown_labels)], shown_labels)
        st.altair_chart(shap_chart, use_container_width=True)
        if "Estancia mínima" in shown_labels:
            st.caption(
                "\\* Su gran dispersión refleja los dos mercados (turístico vs temporada). Más detalle en **Modelo**."
            )

    with st.container(border=True, key="section_analisis_interaccion"):
        st.markdown("#### :material/grid_on: Interacción entre dos variables")
        st.caption(
            "Precio medio predicho para cada combinación de valores de las dos variables elegidas. Si el color "
            "cambia igual de forma vertical en todas las columnas, las variables no interactúan: el efecto de una "
            "no depende de la otra. Si el patrón cambia de una columna a otra, sí hay interacción."
        )

        with st.expander("Ranking: combinaciones con más interacción (para no ir probando a ciegas)"):
            st.caption(
                "Calculado una vez para las 378 combinaciones posibles de las 28 variables. La Correlación "
                "convierte el aviso de \"cuidado con variables relacionadas\" en un dato, el Diagnóstico dice si el "
                "índice alto parece genuino o ambiguo, y Posible explicación es una hipótesis de negocio de por qué "
                "tendría sentido que se combinen (cuando no la hay, lo digo: puede ser solo ruido del modelo)."
            )
            interaction_scores = load_interaction_scores()
            top_df = pd.DataFrame(interaction_scores[:15])
            top_df["correlation"] = [
                pdp_pair_correlation(X_test_matrix, row.a, row.b) for row in top_df.itertuples()
            ]
            top_df["diagnosis"] = [
                interaction_diagnosis(row.score, row.correlation) for row in top_df.itertuples()
            ]
            top_df["business_note"] = [
                interaction_business_note(row.a, row.b) for row in top_df.itertuples()
            ]
            st.dataframe(
                top_df.rename(columns={
                    "a": "Variable A", "b": "Variable B", "score": "Índice",
                    "correlation": "Correlación", "diagnosis": "Diagnóstico",
                    "business_note": "Posible explicación",
                }),
                column_config={
                    "Índice": st.column_config.NumberColumn(format="%.3f"),
                    "Correlación": st.column_config.NumberColumn(format="%.2f"),
                    "Diagnóstico": st.column_config.TextColumn(width="medium"),
                    "Posible explicación": st.column_config.TextColumn(width="large"),
                },
                hide_index=True,
            )
            st.caption(
                "Diagnóstico: heurística orientativa (mismo criterio que el aviso bajo el heatmap), no un test "
                "estadístico formal. Elige dos de esta lista en los desplegables de abajo para ver el mapa de calor completo."
            )

        group_names = list(PDP_GROUPS.keys())
        icol1, icol2 = st.columns(2)
        with icol1:
            inter_a = st.selectbox("Variable en eje X", group_names, index=0, key="inter_a")
        with icol2:
            options_b = [g for g in group_names if g != inter_a]
            inter_b = st.selectbox("Variable en eje Y", options_b, index=0, key="inter_b")

        grid_df, order_a, order_b = pdp_surface(model, X_test_matrix, inter_a, inter_b)
        heatmap = (
            alt.Chart(grid_df)
            .mark_rect()
            .encode(
                x=alt.X("a:O", title=inter_a, sort=order_a),
                y=alt.Y("b:O", title=inter_b, sort=order_b),
                color=alt.Color(
                    "price:Q", title="Precio medio (€)",
                    scale=alt.Scale(scheme="viridis"),
                ),
                tooltip=[
                    alt.Tooltip("a:N", title=inter_a),
                    alt.Tooltip("b:N", title=inter_b),
                    alt.Tooltip("price:Q", title="Precio medio", format=".0f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(heatmap, use_container_width=True)

        ratio = interaction_strength(grid_df)
        if ratio > 0.25:
            st.warning(
                f"**Interacción fuerte** entre estas dos variables (índice {ratio:.2f}): el efecto de una "
                f"cambia claramente según el valor de la otra. El patrón de la curva individual de arriba no "
                f"cuenta toda la historia aquí."
            )
        elif ratio > 0.1:
            st.info(f"**Interacción moderada** (índice {ratio:.2f}): hay algo de dependencia entre ambas, pero no domina el resultado.")
        else:
            st.success(f"**Poca o ninguna interacción** (índice {ratio:.2f}): cada variable actúa de forma bastante independiente de la otra.")
        st.caption(
            "Índice: cuánto se aleja la superficie real de la que habría si cada variable sumara su efecto por "
            "separado, sin combinarse (0 = totalmente independientes, más alto = más interacción). Es una "
            "heurística orientativa, no un test estadístico formal."
        )


def page_formulario():
    st.header("Características del anuncio")

    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        with st.container(border=True, key="section_form_ubicacion"):
            st.markdown("#### :material/location_on: Ubicación")
            dist_col, neigh_col = st.columns(2)
            with dist_col:
                district = st.selectbox(
                    "Distrito", DISTRICTS, index=DISTRICTS.index("Eixample"),
                    help=(
                        "Distrito de Barcelona donde está el alojamiento. Es una de las variables con más peso "
                        "en el precio:\n\n"
                        "- **Ciutat Vella**: casco antiguo, muy turístico\n"
                        "- **Eixample**: zona comercial planificada, muy demandada\n"
                        "- **Gràcia**: barrio bohemio, cerca del centro\n"
                        "- **Sants-Montjuïc**: mixto, cerca de Montjuïc\n"
                        "- **Les Corts**: residencial, zona universitaria\n"
                        "- **Sarrià-Sant Gervasi**: residencial de nivel alto\n"
                        "- **Sant Martí**: playa y Poblenou, en expansión\n"
                        "- **Sant Andreu**: residencial, tranquilo\n"
                        "- **Horta-Guinardó**: residencial, pocos anuncios\n"
                        "- **Nou Barris**: residencial, el más económico"
                    ),
                )
            with neigh_col:
                neighbourhood = st.selectbox(
                    "Barrio", NEIGHBOURHOODS_BY_DISTRICT[district],
                    help="Barrio dentro del distrito elegido. El modelo usa el precio medio histórico de cada barrio como variable.",
                )
            st.caption(":material/pin_drop: Ajustar coordenadas exactas")
            stats = NEIGHBOURHOOD_STATS[neighbourhood]
            if (
                st.session_state.get("coord_neighbourhood") != neighbourhood
                or "coord_lat" not in st.session_state
                or "coord_lon" not in st.session_state
            ):
                # "coord_lat"/"coord_lon" son claves de widgets (los number_input de abajo):
                # Streamlit las elimina de session_state en cuanto sales de esta página, así
                # que hay que volver a poblarlas al entrar, no solo cuando cambia el barrio.
                st.session_state["coord_neighbourhood"] = neighbourhood
                st.session_state["coord_lat"] = stats["lat"]
                st.session_state["coord_lon"] = stats["lon"]

            map_col, coord_col = st.columns([2, 1])
            with map_col:
                coord_map = folium.Map(
                    location=[st.session_state["coord_lat"], st.session_state["coord_lon"]],
                    zoom_start=15,
                    tiles="OpenStreetMap",
                )
                folium.Marker(
                    [st.session_state["coord_lat"], st.session_state["coord_lon"]],
                    icon=folium.Icon(color="red", icon="home"),
                ).add_to(coord_map)
                map_click = st_folium(
                    coord_map,
                    height=140,
                    use_container_width=True,
                    key="coord_map",
                    returned_objects=["last_clicked"],
                )

            if map_click and map_click.get("last_clicked"):
                new_lat = map_click["last_clicked"]["lat"]
                new_lon = map_click["last_clicked"]["lng"]
                if (round(new_lat, 6), round(new_lon, 6)) != (
                    round(st.session_state["coord_lat"], 6),
                    round(st.session_state["coord_lon"], 6),
                ):
                    st.session_state["coord_lat"] = new_lat
                    st.session_state["coord_lon"] = new_lon
                    st.rerun()

            with coord_col:
                latitude = st.number_input(
                    "Latitud", key="coord_lat", format="%.6f",
                    help="Por defecto, el centro del barrio elegido. Ajústala escribiendo o haciendo clic en el mapa.",
                )
                longitude = st.number_input(
                    "Longitud", key="coord_lon", format="%.6f",
                    help="Por defecto, el centro del barrio elegido. Ajústala escribiendo o haciendo clic en el mapa.",
                )

        with st.container(border=True, key="section_form_sobreti"):
            st.markdown("#### :material/person: Sobre ti como anfitrión")
            r1c1, r1c2 = st.columns(2, vertical_alignment="center")
            with r1c1:
                host_is_superhost = st.toggle(
                    "Soy superhost", value=False,
                    help="Distinción de Airbnb para anfitriones con muy buenas valoraciones, alta tasa de respuesta y pocas cancelaciones.",
                )
            with r1c2:
                calculated_host_listings_count = st.number_input(
                    "Anuncios que gestionas", min_value=1, max_value=500, value=1,
                    help="Número total de anuncios que gestionas en Airbnb, incluido este. Los anfitriones con muchos anuncios suelen ser gestores profesionales.",
                )
            r2c1, r2c2 = st.columns(2, vertical_alignment="center")
            with r2c1:
                host_has_profile_pic = st.toggle(
                    "Tengo foto de perfil", value=True,
                    help="Si tu perfil de anfitrión muestra una foto visible. Genera más confianza en los huéspedes.",
                )
            with r2c2:
                host_tenure_years = st.number_input(
                    "Años como anfitrión", min_value=0.0, max_value=20.0, value=0.0, step=0.5,
                    help="Tiempo que llevas publicando anuncios en Airbnb.",
                )
            r3c1, r3c2 = st.columns(2, vertical_alignment="center")
            with r3c1:
                host_identity_verified = st.toggle(
                    "Identidad verificada", value=True,
                    help="Si has verificado tu identidad ante Airbnb (documento oficial, teléfono, email...).",
                )
            with r3c2:
                host_user_tenure_years = st.number_input(
                    "Años con cuenta", min_value=0.0, max_value=25.0,
                    value=max(host_tenure_years, 0.5), step=0.5,
                    help="Años con cuenta en la plataforma: tiempo que llevas registrado en Airbnb, aunque sea sin publicar anuncios.",
                )

    with col_right:
        with st.container(border=True, key="section_form_alojamiento"):
            st.markdown("#### :material/apartment: Alojamiento")
            room_type = st.selectbox(
                "Tipo de alojamiento", ROOM_TYPES,
                help=(
                    "Cómo se comparte el espacio con otros huéspedes:\n\n"
                    "- **Vivienda entera**: el huésped tiene todo el piso para él\n"
                    "- **Habitación privada**: habitación propia dentro de una vivienda compartida\n"
                    "- **Habitación compartida**: comparte también la habitación con otros huéspedes\n"
                    "- **Habitación de hotel**: gestionado como un hotel u hostal"
                ),
            )
            property_type = st.selectbox(
                "Tipo de propiedad", PROPERTY_TYPES,
                help=(
                    "Tipo de inmueble:\n\n"
                    "- **Entire rental unit**: piso de alquiler estándar\n"
                    "- **Entire loft / Entire home / Entire condo**: loft, casa o apartamento en un edificio con servicios\n"
                    "- **Entire serviced apartment**: apartamento con servicios tipo hotel\n"
                    "- **Room in hotel / Room in boutique hotel**: habitación gestionada profesionalmente\n"
                    "- **Private room in...**: habitación privada dentro de ese tipo de vivienda\n"
                    "- **Shared room in hostel**: cama u habitación compartida en un hostel\n"
                    "- **Other**: categorías poco frecuentes, agrupadas"
                ),
            )
            c1, c2 = st.columns(2)
            with c1:
                accommodates = st.number_input(
                    "Huéspedes", min_value=1, max_value=16, value=4,
                    help="Número máximo de personas que pueden alojarse. Una de las variables con más peso en el precio.",
                )
                bedrooms = st.number_input(
                    "Dormitorios", min_value=0, max_value=20, value=1,
                    help="Número de dormitorios del alojamiento.",
                )
                beds = st.number_input(
                    "Camas", min_value=1, max_value=20, value=2,
                    help="Número de camas disponibles (puede diferir de los dormitorios, por ejemplo con literas o sofás cama).",
                )
                bathrooms = st.number_input(
                    "Baños", min_value=0.0, max_value=10.0, value=1.0, step=0.5,
                    help="Número de baños. Se admiten medios baños (0.5) para aseos sin ducha ni bañera.",
                )
            with c2:
                minimum_nights = st.number_input(
                    "Estancia mínima (noches)", min_value=1, max_value=365, value=2,
                    help="Noches mínimas por reserva. Es la variable que más influye en el precio: por encima de "
                    "30 noches el anuncio deja de considerarse turístico, y compite con el alquiler residencial "
                    "normal en vez de con hoteles.",
                )
                st.caption(market_label(minimum_nights))
                maximum_nights = st.number_input(
                    "Estancia máxima (noches)", min_value=1, max_value=1125, value=365,
                    help="Número máximo de noches que se puede reservar de una vez.",
                )
                availability_365 = st.number_input(
                    "Días disponibles al año", min_value=0, max_value=365, value=300,
                    help="Días del año que el anuncio está disponible para reservar. Menor disponibilidad puede indicar un uso más ocasional.",
                )
                with st.container(height=58, border=False, vertical_alignment="center"):
                    has_license = st.toggle(
                        "Tiene licencia turística", value=True,
                        help="Indica si el anuncio tiene número de licencia turística registrado, requisito legal en Barcelona para alquileres de corta duración.",
                    )

        with st.container(border=True, key="section_form_historial"):
            st.markdown("#### :material/history: Historial del anuncio")
            is_new_listing = st.toggle(
                "Es un anuncio nuevo, sin reseñas", value=True,
                help="Actívalo si el anuncio es nuevo y todavía no tiene reseñas ni historial de reservas.",
            )
            review_inputs = {}
            if not is_new_listing:
                c1, c2 = st.columns(2)
                with c1:
                    review_inputs["number_of_reviews"] = st.number_input(
                        "Número de reseñas", min_value=0, value=10,
                        help="Total de reseñas recibidas desde la publicación del anuncio.",
                    )
                    review_inputs["reviews_per_month"] = st.number_input(
                        "Reseñas al mes", min_value=0.0, value=1.0, step=0.1,
                        help="Media de reseñas recibidas por mes, una forma indirecta de estimar la frecuencia de reservas.",
                    )
                    review_inputs["listing_age_days"] = st.number_input(
                        "Días desde la publicación", min_value=0, value=365,
                        help="Antigüedad del anuncio en días.",
                    )
                with c2:
                    review_inputs["number_of_reviews_ltm"] = st.number_input(
                        "Reseñas último año", min_value=0, value=5,
                        help="Reseñas recibidas en los últimos 12 meses.",
                    )
                    review_inputs["review_scores_rating"] = st.slider(
                        "Valoración media", 0.0, 5.0, 4.8,
                        help="Puntuación media de las reseñas, de 0 a 5.",
                    )
                    review_inputs["days_since_last_review"] = st.number_input(
                        "Días desde la última reseña", min_value=0, value=30,
                        help="Cuántos días han pasado desde la reseña más reciente.",
                    )
                review_inputs["estimated_occupancy_l365d"] = st.number_input(
                    "Noches ocupadas último año (estimado)", min_value=0, max_value=365, value=100,
                    help="Estimación de cuántas noches se ha reservado el alojamiento en el último año.",
                )
            else:
                st.caption("Sin reseñas, disponibilidad ni historial todavía: el modelo lo tiene en cuenta como tal.")

    host_entire_homes_ratio = 1.0 if room_type == "Entire home/apt" else 0.0

    current_inputs = {
        "district": district,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "property_type": property_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "bathrooms": bathrooms,
        "minimum_nights": minimum_nights,
        "maximum_nights": maximum_nights,
        "availability_365": availability_365,
        "has_license": has_license,
        "host_is_superhost": host_is_superhost,
        "host_has_profile_pic": host_has_profile_pic,
        "host_identity_verified": host_identity_verified,
        "calculated_host_listings_count": calculated_host_listings_count,
        "host_tenure_years": host_tenure_years,
        "host_user_tenure_years": host_user_tenure_years,
        "host_entire_homes_ratio": host_entire_homes_ratio,
        "is_new_listing": is_new_listing,
        "latitude": latitude,
        "longitude": longitude,
        **review_inputs,
    }

    st.session_state["live_inputs"] = current_inputs
    confirmed_inputs = st.session_state.get("confirmed_inputs", DEFAULT_INPUTS)

    col_btn, col_status = st.columns([1, 2.2], vertical_alignment="center")
    with col_btn:
        clicked = st.button(":material/bolt: Predecir precio", type="primary", use_container_width=True)
    if clicked:
        st.session_state["confirmed_inputs"] = current_inputs
        confirmed_inputs = current_inputs
        st.switch_page(page_prediccion_p)
    with col_status:
        if current_inputs != confirmed_inputs:
            st.warning("Has cambiado el formulario: pulsa **Predecir precio** para actualizar el resultado.")
        else:
            st.success("Resultado actualizado. Ve a **Predicción** para verlo, o a **Explorar mapa** para comparar con anuncios reales.")


inputs = st.session_state.get("confirmed_inputs", DEFAULT_INPUTS)

district = inputs["district"]
neighbourhood = inputs["neighbourhood"]
room_type = inputs["room_type"]
accommodates = inputs["accommodates"]
bedrooms = inputs["bedrooms"]
bathrooms = inputs["bathrooms"]
minimum_nights = inputs["minimum_nights"]
latitude = inputs["latitude"]
longitude = inputs["longitude"]

feature_row = build_features(inputs)
X = pd.DataFrame([feature_row])[FEATURE_COLS]
for c in X.select_dtypes(include="bool").columns:
    X[c] = X[c].astype(int)

pred_log = model.predict(X)[0]
pred = float(np.expm1(pred_log))

base_mae = METADATA["test_metrics"]["MAE"]
flags = confidence_flags(inputs)
multiplier = 2.5 if flags else 1.0
low = max(0.0, pred - base_mae * multiplier)
high = pred + base_mae * multiplier


def page_prediccion():
    st.header("Precio recomendado")
    st.caption(f"Anuncio configurado: {room_type} · {accommodates} huéspedes · {neighbourhood} ({district}).")
    if st.session_state.get("live_inputs", inputs) != inputs:
        st.info(
            "Has cambiado el formulario desde el último cálculo. Ve a **Formulario** y pulsa "
            "**Predecir precio** para ver el resultado actualizado."
        )

    def fmt_whatif_value(v):
        if isinstance(v, bool):
            return "Sí" if v else "No"
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    def whatif_options_and_current(key):
        if key in WHATIF_NUMERIC_VALUES:
            current = inputs.get(key, REVIEW_DEFAULTS.get(key))
            return sorted(set(WHATIF_NUMERIC_VALUES[key]) | {current}), current
        if key == "district":
            return DISTRICTS, district
        if key == "neighbourhood":
            return sorted(NEIGHBOURHOOD_STATS.keys()), neighbourhood
        if key == "room_type":
            return ROOM_TYPES, room_type
        if key == "property_type":
            return PROPERTY_TYPES, inputs["property_type"]
        return [True, False], inputs.get(key, True)

    # Se lee el estado de la simulación "qué pasaría si" (sección de la derecha) antes de
    # pintar el precio y el waterfall, para que ambos reflejen el valor simulado si lo hay.
    whatif_labels_list = list(WHATIF_LABELS.values())
    whatif_choice = st.session_state.get("whatif_choice", whatif_labels_list[0])
    whatif_key = {v: k for k, v in WHATIF_LABELS.items()}[whatif_choice]
    whatif_all_options, whatif_real_val = whatif_options_and_current(whatif_key)
    sim_val_key = f"whatif_val_{whatif_key}"
    reset_flag_key = f"whatif_reset_flag_{whatif_key}"
    if st.session_state.get(reset_flag_key):
        # El botón "Restablecer" solo puede marcar esta bandera (no puede tocar
        # sim_val_key directamente: ese widget ya se instanció en el run anterior).
        # Aquí, al principio del run siguiente y antes de crear el widget, sí se puede.
        st.session_state[sim_val_key] = whatif_real_val
        st.session_state[reset_flag_key] = False
    sim_val = st.session_state.get(sim_val_key, whatif_real_val)
    simulating = sim_val != whatif_real_val

    if simulating:
        sim_inputs = whatif_apply(inputs, whatif_key, sim_val)
        sim_row = build_features(sim_inputs)
        X_disp = pd.DataFrame([sim_row])[FEATURE_COLS]
        for c in X_disp.select_dtypes(include="bool").columns:
            X_disp[c] = X_disp[c].astype(int)
        pred_disp = float(np.expm1(model.predict(X_disp)[0]))
        flags_disp = confidence_flags(sim_inputs)
    else:
        X_disp, pred_disp, flags_disp = X, pred, flags
    multiplier_disp = 2.5 if flags_disp else 1.0
    low_disp = max(0.0, pred_disp - base_mae * multiplier_disp)
    high_disp = pred_disp + base_mae * multiplier_disp

    with st.container(border=True, key="section_prediccion_resumen"):
        conf_text = f"⚠️ {len(flags_disp)} señal(es) de confianza" if flags_disp else "✅ Confianza alta"
        st.markdown("#### :material/payments: Precio estimado")
        price_col, conf_col = st.columns([1, 1.6], vertical_alignment="center")
        with price_col:
            st.markdown(f"# {pred_disp:,.0f} €")
            st.caption(f"Rango: {low_disp:,.0f}–{high_disp:,.0f} €")
        with conf_col:
            # Un solo aviso visible: el recuento de señales y el detalle de cada una en
            # la misma caja (antes el detalle estaba escondido en un expander aparte).
            if flags_disp:
                st.warning("\n\n".join([conf_text] + [f"- {f}" for f in flags_disp]))
            else:
                st.success(conf_text)
        if simulating:
            st.caption(
                f"🔮 Simulando **{whatif_choice} = {fmt_whatif_value(sim_val)}** (real: "
                f"{fmt_whatif_value(whatif_real_val)}) — no cambia lo guardado en el Formulario."
            )

    col_shap, col_whatif = st.columns([1.2, 1])
    with col_shap:
        with st.container(border=True, key="section_prediccion_shap"):
            st.markdown("#### :material/insights: Por qué este precio")
            explainer = get_shap_explainer(model)
            shap_exp = explainer(X_disp)
            base_value = float(shap_exp.base_values[0])
            contributions = pd.Series(shap_exp.values[0], index=FEATURE_COLS)
            group_contrib = contributions.groupby(contributions.index.map(FEATURE_TO_GROUP)).sum()
            group_contrib = group_contrib.reindex(group_contrib.abs().sort_values(ascending=False).index)

            display_inputs = sim_inputs if simulating else inputs
            onehot_selected = {
                "Distrito": display_inputs["district"],
                "Tipo de alojamiento": display_inputs["room_type"],
                "Tipo de propiedad": display_inputs["property_type"],
            }
            def display_label(label):
                selected = onehot_selected.get(label)
                return f"{label}: {selected}" if selected else label

            def group_value_text(label):
                if label == "Latitud":
                    return f"{display_inputs['latitude']:.5f}"
                if label == "Longitud":
                    return f"{display_inputs['longitude']:.5f}"
                spec = PDP_GROUPS.get(label)
                if spec is None:
                    return ""
                kind = spec["kind"]
                if kind == "numeric":
                    val = display_inputs.get(spec["sample_key"], REVIEW_DEFAULTS.get(spec["sample_key"]))
                    return "" if val is None or pd.isna(val) else f"{val:g}"
                if kind == "boolean":
                    return "Sí" if display_inputs.get(spec["col"]) else "No"
                if kind == "onehot":
                    return onehot_selected.get(label, "")
                if kind == "host_scale":
                    val = display_inputs.get("calculated_host_listings_count")
                    return "" if val is None else f"{val:g}"
                return ""

            top_n_wf = 10
            show_all_wf = st.checkbox(f"Ver las {len(group_contrib)} variables", key="show_all_wf")
            if show_all_wf or len(group_contrib) <= top_n_wf:
                steps = list(group_contrib.items())
            else:
                steps = list(group_contrib.head(top_n_wf).items())
                other_sum = group_contrib.iloc[top_n_wf:].sum()
                steps.append((f"Otras {len(group_contrib) - top_n_wf} variables", other_sum))

            # What-if: contribución SHAP de la variable simulada bajo los datos REALES
            # (sin simular), para poder comparar antes/después en su misma fila. Coste
            # despreciable (~1ms una fila más sobre el mismo explainer ya cacheado).
            changed_group = WHATIF_KEY_TO_GROUP.get(whatif_key) if simulating else None
            group_contrib_before = None
            if simulating:
                shap_exp_before = explainer(X)
                contributions_before = pd.Series(shap_exp_before.values[0], index=FEATURE_COLS)
                group_contrib_before = contributions_before.groupby(
                    contributions_before.index.map(FEATURE_TO_GROUP)
                ).sum()

            cum = base_value
            level = np.exp(cum) - 1.0
            rows = [{
                "label": "Punto de partida del modelo",
                "start": 0.0, "end": level, "kind": "punto de partida",
                "etiqueta": f"{level:,.0f} €",
                "color_etiqueta": "#6B6862",
                "tip1": f"Punto de partida del modelo: {level:,.0f} €",
                "tip2": "",
            }]
            for label, val in steps:
                before = level
                cum += val
                level = np.exp(cum) - 1.0
                delta = level - before
                pct = (delta / before * 100) if before else 0.0
                value_text = group_value_text(label)
                # Estancia mínima usa el mismo color que el resto (sube/baja); se marca solo
                # con un asterisco, explicado en la nota al pie del gráfico, porque en gran
                # parte indica a qué mercado pertenece el anuncio (turístico/temporada).
                row_label = display_label(label)
                if label == "Estancia mínima":
                    row_label += " *"
                # El tooltip parte del label crudo, no de display_label: para las one-hot
                # este ya incluye la categoría elegida ("Distrito: Eixample") y volver a
                # pegarle el valor la repetiría.
                wf_row = {
                    "label": row_label,
                    "start": min(before, level), "end": max(before, level),
                    "kind": "sube" if delta >= 0 else "baja",
                    "etiqueta": f"{delta:+,.0f} €".replace("-", "−"),
                    "color_etiqueta": GREEN if delta >= 0 else RED,
                    "tip1": f"{label}: {value_text}" if value_text else label,
                    "tip2": f"{before:,.0f} € → {level:,.0f} €",
                    "tip3": f"{pct:+,.0f} %".replace("-", "−"),
                }
                # Contorno "antes" solo en la fila de la variable que se está simulando:
                # mismo punto de partida ("before"), llegando hasta donde habría llegado
                # con el valor real de esa variable en vez del simulado.
                if group_contrib_before is not None and label == changed_group:
                    old_group_val = group_contrib_before.get(label, 0.0)
                    alt_cum = (cum - val) + old_group_val
                    alt_level = np.exp(alt_cum) - 1.0
                    old_delta = alt_level - before
                    wf_row["etiqueta"] = (
                        f"antes {old_delta:+,.0f} → {delta:+,.0f} €"
                    ).replace("-", "−")
                    wf_row["before_start"] = min(before, alt_level)
                    wf_row["before_end"] = max(before, alt_level)
                    wf_row["before_tip"] = (
                        f"Sin simular: {before:,.0f} € → {alt_level:,.0f} € ({old_delta:+,.0f} €)"
                    ).replace("-", "−")
                rows.append(wf_row)
            total_row = {
                "label": "Precio final", "start": 0.0, "end": level, "kind": "total",
                "etiqueta": f"{level:,.0f} €",
                "color_etiqueta": WF_TOTAL,
                "tip1": f"Precio final: {level:,.0f} €",
                "tip2": "",
            }
            if simulating:
                total_row["before_start"] = 0.0
                total_row["before_end"] = pred
                total_row["before_tip"] = f"Sin simular: {pred:,.0f} €"
            rows.append(total_row)

            if simulating:
                diff = pred_disp - pred
                diff_color = GREEN if diff >= 0 else RED
                diff_str = f"{diff:+,.0f} €".replace("-", "−")
                st.markdown(
                    f"**Precio final: {pred:,.0f} € → {pred_disp:,.0f} € "
                    f"(<span style='color:{diff_color}'>{diff_str}</span>)**",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"Precio final: {pred:,.0f} € (sin cambios respecto a tu formulario).")

            chart = chart_waterfall(pd.DataFrame(rows), x_title="Precio (€)")
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "\\* Refleja sobre todo si tu anuncio es turístico o de temporada. "
                "Más detalle en **Modelo**."
            )

    with col_whatif:
        with st.container(border=True, key="section_prediccion_whatif"):
            st.markdown("#### :material/tune: ¿Qué pasaría si cambio...?")
            st.caption("Prueba un valor: el precio y el desglose se recalculan sin tocar tu formulario.")

            wcol1, wcol2 = st.columns(2)
            with wcol1:
                whatif_choice = st.selectbox("Variable a explorar", whatif_labels_list, key="whatif_choice")
            whatif_key = {v: k for k, v in WHATIF_LABELS.items()}[whatif_choice]
            options, current_val = whatif_options_and_current(whatif_key)
            this_sim_key = f"whatif_val_{whatif_key}"
            if this_sim_key not in st.session_state:
                st.session_state[this_sim_key] = current_val
            is_numeric_whatif = whatif_key in WHATIF_NUMERIC_VALUES

            with wcol2:
                if is_numeric_whatif:
                    sim_val = st.select_slider("Simula un valor", options=options, key=this_sim_key)
                else:
                    sim_val = st.selectbox(
                        "Simula un valor", options, format_func=fmt_whatif_value, key=this_sim_key,
                    )

            if is_numeric_whatif:
                prices = [whatif_predict(model, inputs, whatif_key, v) for v in options]
                chart = chart_line_highlight(options, prices, current_val, x_title=whatif_choice)
                st.altair_chart(chart, use_container_width=True)
            else:
                prices_all = [whatif_predict(model, inputs, whatif_key, o) for o in options]
                order = sorted(range(len(options)), key=lambda i: prices_all[i], reverse=True)
                # La ventana se centra en el valor SIMULADO (el del desplegable), no en el
                # guardado en el formulario: si no, al simular un valor lejano del real, la
                # ventana se queda donde estaba y el simulado puede quedar fuera de pantalla.
                sim_rank = order.index(options.index(sim_val))

                top_n_cat = 12
                if len(options) > top_n_cat:
                    show_all_cat = st.checkbox(
                        f"Ver las {len(options)} opciones", key=f"whatif_expand_{whatif_key}",
                    )
                else:
                    show_all_cat = True

                if show_all_cat:
                    shown_idx = order
                else:
                    # Ventana de top_n_cat opciones centrada en el rango (por precio) del
                    # valor simulado, para que siempre se vea sin tener que desplegar todo.
                    half = top_n_cat // 2
                    lo = max(0, min(sim_rank - half, len(order) - top_n_cat))
                    shown_idx = order[lo:lo + top_n_cat]

                display_labels = [fmt_whatif_value(options[i]) for i in shown_idx]
                shown_prices = [prices_all[i] for i in shown_idx]
                sim_label = fmt_whatif_value(sim_val)

                # El valor real guardado no se dibuja como barra (podría caer fuera de la
                # ventana visible): se muestra como texto encima, para comparar contra la
                # barra naranja del simulado sin riesgo de que algo quede oculto.
                if sim_val != current_val:
                    real_price = prices_all[options.index(current_val)]
                    st.caption(
                        f"Tu {whatif_choice.lower()} real ({fmt_whatif_value(current_val)}): "
                        f"{real_price:,.0f} €"
                    )

                chart = chart_hbar_highlight(display_labels, shown_prices, sim_label, x_title="Precio predicho (€)")
                st.altair_chart(chart, use_container_width=True)

            if sim_val != current_val:
                if st.button("Restablecer al valor real", key=f"whatif_reset_btn_{whatif_key}"):
                    st.session_state[reset_flag_key] = True
                    st.rerun()


AMBER = "#BA7517"  # veredicto "fuera del rango típico" en Comparar (GREEN = dentro del rango)


def page_comparar():
    st.header("Comparar")
    st.caption(
        "¿Tu precio encaja con lo que se cobra por algo como lo tuyo, cerca de ti? Usa el anuncio confirmado "
        "en tu formulario."
    )

    veredicto_container = st.container(border=True, key="section_comparar_veredicto")

    with st.container(border=True, key="section_comparar_mapa"):
        st.markdown("#### :material/map: Dónde están estos anuncios")

        col_map, col_filters = st.columns([2.2, 1.3])

        with col_filters:
            st.markdown("**Parecido a lo mío**")
            # Igual que en el filtro de tipo de alojamiento de Explorar mapa: st.pills no
            # renderiza bien un solo elemento por defecto si se usa selectbox, así que se
            # mantiene el mismo patrón de pills multi-selección, empezando solo con el tuyo.
            room_type_choices = st.pills(
                "Tipo de alojamiento",
                list(ROOM_TYPE_ICONS.values()),
                selection_mode="multi",
                default=[ROOM_TYPE_ICONS[room_type]],
            )
            room_types_filter = [ROOM_TYPE_LABELS[c] for c in (room_type_choices or [])] or ROOM_TYPES

            acc_lo, acc_hi = st.slider(
                "Huéspedes", min_value=1, max_value=16,
                value=(max(1, accommodates - 1), min(16, accommodates + 1)),
                help="Empieza en un margen estrecho (±1) sobre tu número de huéspedes; ábrelo para ampliar la comparación.",
            )
            # "Parecido" en estancia mínima significa "mismo mercado" (turístico <=30 vs
            # temporada >=31), no un margen de noches: dos anuncios a 2 y 25 noches son
            # ambos turísticos y comparables; uno a 2 y otro a 45 no lo son (ver Modelo).
            default_mn_market = (1, MARKET_CUTOFF) if minimum_nights <= MARKET_CUTOFF else (MARKET_CUTOFF + 1, 365)
            mn_lo, mn_hi = st.slider(
                "Estancia mínima (noches)", min_value=1, max_value=365,
                value=default_mn_market,
                help="Empieza acotado a tu mismo mercado (turístico ≤30 noches o temporada ≥31); ábrelo para ampliar.",
            )

            radius = st.radio(
                "Alcance",
                ["Tu barrio", "Tu distrito", "Toda la ciudad"],
                index=1,
                horizontal=True,
                help="El punto de partida es tu distrito: suficientes comparables sin perder lo local. "
                "Estréchalo a tu barrio o ábrelo a toda la ciudad.",
            )

        similar = find_similar_comparables(
            room_types=room_types_filter, acc_range=(acc_lo, acc_hi), mn_range=(mn_lo, mn_hi), radius=radius,
        )
        n_similar = len(similar)

    with veredicto_container:
        st.markdown("#### :material/fact_check: Tu precio frente al mercado")

        if n_similar < 10:
            st.warning(
                f"Solo {n_similar} anuncios similares en tu zona — amplía el radio para una comparación más fiable."
            )

        if n_similar == 0:
            st.caption("No hay anuncios similares en tu distrito todavía para comparar tu precio con el mercado.")
        else:
            p25, p75 = similar["price"].quantile([0.25, 0.75])
            if pred < p25:
                verdict, color = "Por debajo de lo típico", AMBER
            elif pred > p75:
                verdict, color = "Por encima de lo típico", AMBER
            else:
                verdict, color = "Dentro del rango habitual", GREEN

            vcol1, vcol2, vcol3 = st.columns(3)
            vcol1.metric("Tu precio", f"{pred:,.0f} €")
            vcol2.metric("Rango típico de similares (25-75%)", f"{p25:,.0f} – {p75:,.0f} €")
            vcol3.metric("Anuncios similares", f"{n_similar}")
            # Insignia y nota a pie fundidas en una sola línea (antes eran dos bloques
            # separados) para ahorrar el alto de un párrafo completo.
            st.markdown(
                f"""<span style="background:{color}1A;color:{color};padding:4px 10px;border-radius:6px;
                font-weight:600;">{verdict}</span> — comparado con {n_similar} anuncios reales en {district},
                mismo tipo, tamaño similar y estancia mínima parecida.""",
                unsafe_allow_html=True,
            )

    listing_point = pd.DataFrame([{
        "latitude": latitude, "longitude": longitude, "price": pred,
        "name": "Tu anuncio (estimado)", "picture_url": "", "room_type": room_type,
        "bedrooms": bedrooms, "bathrooms": bathrooms, "minimum_nights": minimum_nights,
        "accommodates": accommodates, "listing_url": "",
    }])

    hover_card_html = """
    <div style="max-width:220px">
      <img src="{picture_url}" onerror="this.style.display='none'"
           style="width:100%;border-radius:6px;display:block;margin-bottom:6px" />
      <b>{name}</b><br/>
      {price:.0f} € / noche · {room_type}<br/>
      {bedrooms} hab · {bathrooms} baños · min. {minimum_nights} noches
    </div>
    """
    # Un único campo "tooltip_html" precalculado por fila/feature en todas las capas
    # (puntos y contorno de zona), en vez de placeholders sueltos: así el mismo tooltip
    # global del Deck sirve tanto para un anuncio como para el nombre de la zona, sin
    # que le falten campos a una de las dos capas.
    listing_point["tooltip_html"] = [hover_card_html.format(**listing_point.iloc[0])]
    if n_similar:
        similar = similar.copy()
        similar["tooltip_html"] = [hover_card_html.format(**row._asdict()) for row in similar.itertuples()]

    # Contorno de la zona activa (barrio o distrito), solo como contexto: gris medio,
    # relleno casi transparente para poder hacer hover en toda el área, sin competir
    # visualmente con los puntos azules ni con el rojo del usuario. En "Toda la ciudad"
    # no hay una zona única que dibujar, así que no se añade ninguna capa.
    zone_layer = None
    if radius == "Tu barrio":
        zone_name = neighbourhood
        zone_features = [
            f for f in load_neighbourhood_geojson()["features"]
            if f["properties"]["neighbourhood"] == neighbourhood
        ]
    elif radius == "Tu distrito":
        zone_name = district
        zone_features = [
            f for f in load_district_geojson()["features"] if f["properties"]["district"] == district
        ]
    else:
        zone_name = None
        zone_features = []

    if zone_features:
        zone_geojson = {
            "type": "FeatureCollection",
            "features": [
                {**f, "properties": {**f["properties"], "tooltip_html": f"<b>{zone_name}</b>"}}
                for f in zone_features
            ],
        }
        zone_layer = pdk.Layer(
            "GeoJsonLayer",
            id="zone-outline",
            data=zone_geojson,
            stroked=True,
            filled=True,
            get_fill_color=[154, 160, 166, 20],
            get_line_color=[110, 115, 120, 220],
            get_line_width=2,
            line_width_min_pixels=2,
            line_width_units="'pixels'",
            pickable=True,
        )

    layers = []
    if zone_layer is not None:
        layers.append(zone_layer)
    if n_similar:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="comparables",
                data=similar,
                get_position="[longitude, latitude]",
                get_radius=8,
                radius_units="'pixels'",
                radius_min_pixels=6,
                radius_max_pixels=14,
                get_fill_color=[100, 149, 237, 210],
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1,
                stroked=True,
                pickable=True,
                auto_highlight=True,
                highlight_color=[255, 215, 0, 230],
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            id="your-listing",
            data=listing_point,
            get_position="[longitude, latitude]",
            get_radius=10,
            radius_units="'pixels'",
            radius_min_pixels=8,
            radius_max_pixels=16,
            get_fill_color=[255, 56, 92, 230],
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            stroked=True,
        )
    )
    view_state = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=13)
    tooltip = {
        "html": "{tooltip_html}",
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "fontSize": "12px",
            "borderRadius": "8px",
            "padding": "8px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.25)",
        },
    }

    with col_map:
        map_state = st.pydeck_chart(
            pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip, map_style="light"),
            on_select="rerun",
            selection_mode="single-object",
            height=335,
            key="comparar_map",
        )
        st.caption(f"{n_similar} anuncios similares cerca de ti (punto rojo = tu anuncio).")

        selected = []
        if map_state and map_state.selection:
            selected = map_state.selection.get("objects", {}).get("comparables", [])

        if selected:
            listing = selected[0]
            card_col1, card_col2 = st.columns([1, 1.5])
            with card_col1:
                if isinstance(listing.get("picture_url"), str) and listing["picture_url"].startswith("http"):
                    st.image(listing["picture_url"], width="stretch")
            with card_col2:
                st.markdown(f"**{listing.get('name', 'Anuncio')}**")
                st.write(f"{listing['price']:.0f} € / noche · {listing.get('room_type', '')}")
                st.write(f"{listing.get('bedrooms', '?')} dormitorios · {listing.get('bathrooms', '?')} baños · {listing.get('accommodates', '?')} huéspedes")
                st.write(f"Estancia mínima: {listing.get('minimum_nights', '?')} noches")
                if listing.get("listing_url"):
                    st.link_button("Ver en Airbnb", listing["listing_url"])
        elif n_similar:
            st.caption("Haz clic en un punto azul del mapa para ver el detalle de ese anuncio.")


MERCADO_METRICS = {
    "Precio medio": {
        "col": "avg_price", "kind": "sequential", "legend_title": "Precio medio (€)",
        "value_fmt": lambda v: f"{v:,.0f} €", "tooltip_fmt": lambda v: f"precio medio {v:,.0f} €",
    },
    "Ocupación media anual": {
        "col": "avg_occupancy", "kind": "sequential", "legend_title": "Ocupación media (noches/año)",
        "value_fmt": lambda v: f"{v:,.0f} noches", "tooltip_fmt": lambda v: f"ocupación media {v:,.0f} noches",
    },
    "Densidad de anuncios": {
        "col": "n_listings", "kind": "sequential", "legend_title": "Nº de anuncios",
        "value_fmt": lambda v: f"{v:,.0f}", "tooltip_fmt": lambda v: f"{v:,.0f} anuncio" + ("" if v == 1 else "s"),
    },
    "Valoración media": {
        "col": "avg_rating", "kind": "sequential", "legend_title": "Valoración media (1-5)",
        "value_fmt": lambda v: f"{v:.2f}", "tooltip_fmt": lambda v: f"valoración media {v:.2f}",
    },
    # Divergente (centro fijo en 0.5) SOLO para variables genuinamente binarias de dos
    # polos. "% de corta estancia" lo es (turístico <=30 noches vs temporada >30). Las
    # demás proporciones (vivienda entera, licencia) resumen una sola categoría sobre el
    # total y no tienen un "polo opuesto" real (vivienda entera tiene 3 alternativas
    # distintas, no 1), así que van con el mismo secuencial que el resto de la página.
    "% de corta estancia (turístico vs temporada)": {
        "col": "pct_corta_estancia", "kind": "diverging", "legend_title": "% de corta estancia",
        "colors": (MARKET_COLOR, GRAY_MUTED, BLUE), "labels": ("Temporada", "Turístico"),
        "tooltip_fmt": lambda v: (
            lambda pct: f"{pct:.0f}% corta estancia · {100 - pct:.0f}% temporada"
        )(round(v * 100)),
        "footnote": "Los dos mercados (turístico vs temporada) se explican en **Modelo**.",
    },
    "% de vivienda entera": {
        "col": "pct_entire_home", "kind": "sequential", "legend_title": "% de vivienda entera",
        "value_fmt": lambda v: f"{v * 100:,.0f} %", "tooltip_fmt": lambda v: f"{v * 100:.0f}% vivienda entera",
    },
    "% con licencia": {
        "col": "pct_license", "kind": "sequential", "legend_title": "% con licencia turística",
        "value_fmt": lambda v: f"{v * 100:,.0f} %", "tooltip_fmt": lambda v: f"{v * 100:.0f}% con licencia",
    },
}


def page_mercado():
    st.header("Mapa del mercado")
    st.caption(
        "Precios y oferta de Airbnb en Barcelona por zonas, a partir de los anuncios reales usados en todo el "
        "resto de la app. No depende de tu formulario ni de ningún anuncio concreto."
    )

    with st.container(border=True, key="section_mercado_mapa"):
        st.markdown("#### :material/public: Mapa por barrio")

        col_map, col_controls = st.columns([2.6, 1])

        with col_controls:
            metric_choice = st.selectbox("Ver por", list(MERCADO_METRICS.keys()), key="mercado_metric")
            metric = MERCADO_METRICS[metric_choice]

            st.divider()
            st.markdown("**Filtrar**")
            room_filter_choice = st.pills(
                "Filtrar por tipo de alojamiento",
                ["Todos"] + list(ROOM_TYPE_ICONS.values()),
                default="Todos",
                key="mercado_room_filter",
            )
            room_type_filter = (
                "Todos" if room_filter_choice in (None, "Todos") else ROOM_TYPE_LABELS[room_filter_choice]
            )

            accommodates_filter = st.slider(
                "Filtrar por nº de huéspedes",
                1, 16, (1, 16),
                key="mercado_accommodates_filter",
            )

            bedrooms_filter = st.slider(
                "Filtrar por nº de dormitorios",
                0, BEDROOMS_SLIDER_MAX, (0, BEDROOMS_SLIDER_MAX),
                key="mercado_bedrooms_filter",
            )
            st.caption(f"{BEDROOMS_SLIDER_MAX} = {BEDROOMS_SLIDER_MAX} o más dormitorios.")

        col = metric["col"]
        agg = neighbourhood_market_agg(
            room_type_filter, accommodates_filter, bedrooms_filter
        ).set_index("neighbourhood")
        geojson = load_neighbourhood_geojson()

        if metric["kind"] == "sequential":
            vmin, vmax = float(agg[col].min()), float(agg[col].max())
            colormap = cm.LinearColormap(colors=[WHATIF_ALT, RED], vmin=vmin, vmax=vmax)
        else:
            # Divergente con el neutro fijado en 0.5, no en la media de los datos: el
            # gris debe significar siempre "repartido 50/50", no "valor típico". Cada
            # métrica divergente trae sus propios colores de polo (metric["colors"]),
            # para no confundir "turístico/temporada" con "habitación/vivienda entera".
            vmin, vmax = 0.0, 1.0
            low_color, mid_color, high_color = metric["colors"]
            colormap = cm.LinearColormap(colors=[low_color, mid_color, high_color], index=[0, 0.5, 1], vmin=0, vmax=1)

        features = []
        for feat in geojson["features"]:
            name = feat["properties"]["neighbourhood"]
            props = dict(feat["properties"])
            # Un barrio puede tener anuncios pero ningún valor válido para esta métrica
            # en concreto (p.ej. "Valoración media" si ninguno tiene reseñas todavía), o
            # quedarse con muy pocos anuncios tras los filtros: en ambos casos se trata
            # como "sin datos", en vez de un 0/NaN que rompa el colormap o una media poco
            # fiable calculada sobre 2-3 casos.
            has_enough = name in agg.index and agg.loc[name, "n_listings"] >= MIN_LISTINGS_PER_BARRIO
            value = agg.loc[name, col] if has_enough else None
            if value is not None and pd.notna(value):
                props["fill_color"] = colormap(float(value))
                props["tooltip_text"] = f"{name} · {metric['tooltip_fmt'](value)}"
            else:
                props["fill_color"] = "#D8D5CC"
                props["tooltip_text"] = f"{name} · sin datos"
            features.append({**feat, "properties": props})
        geojson_colored = {"type": "FeatureCollection", "features": features}

        fmap = folium.Map(location=[41.3925, 2.1665], zoom_start=12, tiles="OpenStreetMap")
        folium.GeoJson(
            geojson_colored,
            style_function=lambda feat: {
                "fillColor": feat["properties"]["fill_color"],
                "color": "#FFFFFF",
                "weight": 1,
                "fillOpacity": 0.85,
            },
            highlight_function=lambda feat: {"weight": 2, "color": "#3A3A37"},
            tooltip=folium.GeoJsonTooltip(fields=["tooltip_text"], aliases=[""], labels=False, sticky=True),
        ).add_to(fmap)

        with col_map:
            st_folium(fmap, use_container_width=True, height=480, returned_objects=[], key="mercado_map")

        with col_controls:
            # Leyenda propia en vez de la de branca (su ancho se estira con el mapa y el
            # texto acaba solapado con la barra de color): control total del layout aquí.
            if metric["kind"] == "sequential":
                st.markdown(
                    f"""
                    <div style="margin-top:4px;">
                      <div style="font-size:12px;color:#6B6862;margin-bottom:4px;">{metric['legend_title']}</div>
                      <div style="height:10px;border-radius:5px;
                                  background:linear-gradient(to right, {WHATIF_ALT}, {RED});"></div>
                      <div style="display:flex;justify-content:space-between;font-size:12px;color:#6B6862;margin-top:2px;">
                        <span>{metric['value_fmt'](vmin)}</span><span>{metric['value_fmt'](vmax)}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                low_color, mid_color, high_color = metric["colors"]
                label_low, label_high = metric["labels"]
                st.markdown(
                    f"""
                    <div style="margin-top:4px;">
                      <div style="font-size:12px;color:#6B6862;margin-bottom:4px;">{metric['legend_title']}</div>
                      <div style="height:10px;border-radius:5px;
                                  background:linear-gradient(to right, {low_color}, {mid_color}, {high_color});"></div>
                      <div style="display:flex;justify-content:space-between;font-size:11px;color:#6B6862;margin-top:2px;">
                        <span>{label_low}</span><span>50/50</span><span>{label_high}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if metric.get("footnote"):
                    st.caption(metric["footnote"])


page_home_p = st.Page(page_home, title="Inicio", icon=":material/home:", default=True)
page_formulario_p = st.Page(page_formulario, title="Formulario", icon=":material/edit_note:")
page_prediccion_p = st.Page(page_prediccion, title="Predicción", icon=":material/payments:")
page_comparar_p = st.Page(page_comparar, title="Comparar", icon=":material/compare:")
page_modelo_p = st.Page(page_modelo, title="Modelo", icon=":material/account_tree:")
page_variables_p = st.Page(page_variables, title="Variables", icon=":material/category:")
page_mercado_p = st.Page(page_mercado, title="Mapa del mercado", icon=":material/public:")
page_analisis_p = st.Page(page_analisis, title="Análisis", icon=":material/query_stats:")

pg = st.navigation({
    "": [page_home_p],
    "Tu anuncio": [page_formulario_p, page_prediccion_p, page_comparar_p],
    "El modelo": [page_modelo_p, page_variables_p, page_analisis_p],
    "Datos": [page_mercado_p],
})
pg.run()
