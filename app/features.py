"""Reconstruye el vector de 66 features que espera price_model.pkl a partir
de un formulario simple, sin obligar al usuario a conocer columnas derivadas
como neighbourhood_price_encoded o distance_to_center_km.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

with open(PROJECT_DIR / "models" / "model_metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURE_COLS = METADATA["feature_cols"]

# Árbol espacial sobre los anuncios reales (mismo universo y radio que
# n_nearby_150m en 02_feature_engineering.ipynb), para poder calcular la densidad
# de un anuncio nuevo/simulado sin tener que recorrer los ~13000 anuncios cada vez.
_EARTH_R_KM = 6371.0
_NEARBY_RADIUS_M = 150
_nearby_coords = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "listings_full_clean.csv", usecols=["latitude", "longitude"]
)
_NEARBY_TREE = BallTree(np.radians(_nearby_coords[["latitude", "longitude"]].to_numpy()), metric="haversine")


def count_nearby(lat, lon, radius_m=_NEARBY_RADIUS_M):
    radius_rad = (radius_m / 1000) / _EARTH_R_KM
    count = _NEARBY_TREE.query_radius(np.radians([[lat, lon]]), r=radius_rad, count_only=True)
    return int(count[0])

with open(APP_DIR / "assets" / "neighbourhood_stats.json", encoding="utf-8") as f:
    NEIGHBOURHOOD_STATS = json.load(f)

DISTRICTS = sorted({v["district"] for v in NEIGHBOURHOOD_STATS.values()})

NEIGHBOURHOODS_BY_DISTRICT = {
    district: sorted(n for n, v in NEIGHBOURHOOD_STATS.items() if v["district"] == district)
    for district in DISTRICTS
}

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

# Mismas 13 categorías que quedaron como columnas propias en 02_feature_engineering.ipynb
# (>=30 apariciones en el dataset); el resto se agrupa en "Other".
PROPERTY_TYPES = [
    "Entire rental unit",
    "Private room in rental unit",
    "Entire serviced apartment",
    "Entire condo",
    "Room in hotel",
    "Private room in hostel",
    "Entire loft",
    "Private room in home",
    "Room in boutique hotel",
    "Entire home",
    "Private room in condo",
    "Shared room in hostel",
    "Private room in bed and breakfast",
    "Other",
]

CITY_CENTER = (41.3874, 2.1686)  # Plaça Catalunya, misma referencia que en 02_feature_engineering.ipynb

FEATURE_LABELS = {
    "minimum_nights": "Estancia mínima (noches)",
    "maximum_nights": "Estancia máxima (noches)",
    "accommodates": "Huéspedes",
    "bedrooms": "Dormitorios",
    "beds": "Camas",
    "bathrooms": "Baños",
    "availability_365": "Días disponibles al año",
    "has_license": "Tiene licencia turística",
    "host_is_superhost": "Es superhost",
    "host_has_profile_pic": "Tiene foto de perfil",
    "host_identity_verified": "Identidad verificada",
    "host_entire_homes_ratio": "% de sus anuncios que son vivienda entera",
    "host_tenure_years": "Años como anfitrión",
    "host_user_tenure_years": "Años con cuenta en la plataforma",
    "calculated_host_listings_count": "Anuncios que gestiona el anfitrión",
    "neighbourhood_price_encoded": "Nivel de precios del barrio",
    "distance_to_center_km": "Distancia al centro (km)",
    "latitude": "Latitud",
    "longitude": "Longitud",
    "has_reviews": "Tiene reseñas",
    "number_of_reviews": "Número de reseñas",
    "number_of_reviews_ltm": "Reseñas último año",
    "reviews_per_month": "Reseñas al mes",
    "review_scores_rating": "Valoración media",
    "listing_age_days": "Antigüedad del anuncio (días)",
    "days_since_last_review": "Días desde la última reseña",
    "estimated_occupancy_l365d": "Noches ocupadas último año",
    "has_ac": "Aire acondicionado",
    "has_pool": "Piscina",
    "has_dishwasher": "Lavavajillas",
    "n_amenities": "Nº de amenities",
    "n_nearby_150m": "Anuncios cercanos (150m)",
}


def pretty_feature_name(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.endswith("_log"):
        base = feature[: -len("_log")]
        return f"{pretty_feature_name(base)} (log)"
    for prefix, category in [
        ("room_type_", "Tipo"),
        ("district_", "Distrito"),
        ("property_type_", "Propiedad"),
        ("host_size_", "Tamaño anfitrión"),
    ]:
        if feature.startswith(prefix):
            return f"{category}: {feature[len(prefix):]}"
    return feature.replace("_", " ").capitalize()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def host_size_tier(listings_count):
    if listings_count <= 1:
        return "individual"
    if listings_count <= 10:
        return "pequeno"
    if listings_count <= 50:
        return "mediano"
    return "grande"


def build_features(inputs: dict):
    """inputs: dict con las claves definidas en app.py (ver sidebar).
    Devuelve un dict alineado con FEATURE_COLS, listo para pasar a pandas.DataFrame.
    """
    neigh_stats = NEIGHBOURHOOD_STATS[inputs["neighbourhood"]]
    lat = inputs.get("latitude") or neigh_stats["lat"]
    lon = inputs.get("longitude") or neigh_stats["lon"]
    distance_to_center_km = haversine_km(lat, lon, *CITY_CENTER)
    n_nearby_150m = count_nearby(lat, lon)

    listings_count = inputs["calculated_host_listings_count"]
    tier = host_size_tier(listings_count)

    is_new_listing = inputs["is_new_listing"]
    if is_new_listing:
        number_of_reviews = 0
        number_of_reviews_ltm = 0
        reviews_per_month = 0.0
        review_scores_rating = float("nan")
        has_reviews = False
        listing_age_days = float("nan")
        days_since_last_review = float("nan")
        estimated_occupancy_l365d = float("nan")
    else:
        number_of_reviews = inputs["number_of_reviews"]
        number_of_reviews_ltm = inputs["number_of_reviews_ltm"]
        reviews_per_month = inputs["reviews_per_month"]
        review_scores_rating = inputs["review_scores_rating"]
        has_reviews = number_of_reviews > 0
        listing_age_days = inputs["listing_age_days"]
        days_since_last_review = inputs["days_since_last_review"]
        estimated_occupancy_l365d = inputs["estimated_occupancy_l365d"]

    row = {
        "host_is_superhost": inputs["host_is_superhost"],
        "host_has_profile_pic": inputs["host_has_profile_pic"],
        "host_identity_verified": inputs["host_identity_verified"],
        "latitude": lat,
        "longitude": lon,
        "accommodates": inputs["accommodates"],
        "bathrooms": inputs["bathrooms"],
        "bedrooms": inputs["bedrooms"],
        "beds": inputs["beds"],
        "minimum_nights": inputs["minimum_nights"],
        "maximum_nights": inputs["maximum_nights"],
        "availability_365": inputs["availability_365"],
        "number_of_reviews": number_of_reviews,
        "number_of_reviews_ltm": number_of_reviews_ltm,
        "estimated_occupancy_l365d": estimated_occupancy_l365d,
        "review_scores_rating": review_scores_rating,
        "calculated_host_listings_count": listings_count,
        "reviews_per_month": reviews_per_month,
        "has_license": inputs["has_license"],
        "number_of_reviews_log": math.log1p(number_of_reviews),
        "number_of_reviews_ltm_log": math.log1p(number_of_reviews_ltm),
        "reviews_per_month_log": math.log1p(reviews_per_month),
        "bedrooms_log": math.log1p(inputs["bedrooms"]),
        "beds_log": math.log1p(inputs["beds"]),
        "bathrooms_log": math.log1p(inputs["bathrooms"]),
        "calculated_host_listings_count_log": math.log1p(listings_count),
        "neighbourhood_price_encoded": neigh_stats["price_encoded"],
        "host_tenure_years": inputs["host_tenure_years"],
        "host_entire_homes_ratio": inputs["host_entire_homes_ratio"],
        "has_reviews": has_reviews,
        "listing_age_days": listing_age_days,
        "days_since_last_review": days_since_last_review,
        "distance_to_center_km": distance_to_center_km,
        "host_user_tenure_years": inputs["host_user_tenure_years"],
        "has_ac": inputs["has_ac"],
        "has_pool": inputs["has_pool"],
        "has_dishwasher": inputs["has_dishwasher"],
        "n_amenities": inputs["n_amenities"],
        "n_nearby_150m": n_nearby_150m,
    }

    for rt in ROOM_TYPES:
        row[f"room_type_{rt}"] = inputs["room_type"] == rt

    for d in DISTRICTS:
        row[f"district_{d}"] = neigh_stats["district"] == d

    for pt in PROPERTY_TYPES:
        row[f"property_type_{pt}"] = inputs["property_type"] == pt

    for t in ["individual", "pequeno", "mediano", "grande"]:
        row[f"host_size_{t}"] = tier == t

    return {col: row[col] for col in FEATURE_COLS}


def confidence_flags(inputs: dict):
    """Señales de baja confianza documentadas en 05_model_evaluation.ipynb:
    anuncios tipo Group Flat (cola alta de precio) y distritos con poca muestra.
    """
    flags = []
    if inputs["bedrooms"] >= 7 or inputs["accommodates"] >= 10:
        flags.append(
            "Este tamaño de alojamiento (Group Flat) es donde el modelo subestima "
            "más el precio en los datos históricos: trata el resultado como orientativo."
        )
    neigh_stats = NEIGHBOURHOOD_STATS[inputs["neighbourhood"]]
    if neigh_stats["district"] in ("Horta-Guinardó", "Nou Barris"):
        flags.append(
            f"{neigh_stats['district']} tiene poca representación en los datos de entrenamiento, "
            "así que la predicción es menos fiable que en distritos con más anuncios."
        )
    if inputs["is_new_listing"]:
        flags.append(
            "Al no tener reseñas todavía, el modelo estima con menos información de la habitual."
        )
    return flags
