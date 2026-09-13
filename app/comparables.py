"""Conjunto único de comparables para la pantalla Comparar: un anuncio del
usuario entra, un conjunto de anuncios reales ordenados por parecido sale.
Veredicto, lista y mapa consumen el resultado de este módulo para no volver
a calcular "los mismos" comparables de formas distintas y contradecirse
entre sí. Sin Streamlit dentro: solo pandas/numpy, para poder testear y
razonar sobre esto sin levantar la app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Si un alcance geográfico se queda con menos comparables que esto, se amplía
# al siguiente (barrio -> distrito -> ciudad) antes de darlo por bueno.
MIN_COMPARABLES = 30

# Como mucho 2 anuncios del mismo anfitrión sobreviven al ordenar por
# proximidad: los datos de Airbnb están llenos de hosts con varias
# propiedades casi idénticas, y dejarlas todas sesga el conjunto hacia ese
# host en vez de reflejar variedad real del mercado.
MAX_POR_ANFITRION = 2

ALCANCES = ("barrio", "distrito", "ciudad")

COL_BARRIO = "neighbourhood_cleansed"
COL_DISTRITO = "neighbourhood_group_cleansed"

# Cajones de estancia mínima: no es un rango libre alrededor del valor del
# usuario porque los anuncios de estancia larga tienen otra lógica de precio
# y contaminarían el rango típico de uno de estancia corta con un valor de
# minimum_nights parecido en número pero de mercado distinto (ver
# MARKET_CUTOFF en app.py para la frontera turístico/temporada, un corte
# distinto y más grueso que este).
CORTA_MAX = 3
MEDIA_MAX = 7


def tramo_estancia(minimum_nights) -> str:
    """'corta' (1-3 noches), 'media' (4-7) o 'larga' (8+). Un valor por
    debajo de 1 (no debería darse en datos reales) cae en 'corta'."""
    if minimum_nights <= CORTA_MAX:
        return "corta"
    if minimum_nights <= MEDIA_MAX:
        return "media"
    return "larga"


def _tramo_estancia_vectorizado(minimum_nights: pd.Series) -> pd.Series:
    condiciones = [minimum_nights <= CORTA_MAX, minimum_nights <= MEDIA_MAX]
    opciones = ["corta", "media"]
    return pd.Series(np.select(condiciones, opciones, default="larga"), index=minimum_nights.index)


@dataclass(frozen=True)
class ResultadoComparables:
    df: pd.DataFrame
    alcance_solicitado: str
    alcance_final: str
    se_amplio: bool
    n_en_alcance_original: int
    insuficiente: bool  # True si ni siquiera a nivel ciudad se llega a MIN_COMPARABLES


def conjunto_comparables(anuncio_usuario: dict, df_zona: pd.DataFrame, alcance: str) -> ResultadoComparables:
    """Filtra df_zona (todos los anuncios de la ciudad activa, ver
    load_comparables en app.py) a los comparables de anuncio_usuario.

    Filtros duros, nunca relajados: mismo room_type, capacidad ±1, mismo
    número de bedrooms y mismo tramo de estancia mínima (ver
    tramo_estancia). El alcance geográfico (barrio/distrito/ciudad) sí se
    amplía como respaldo si el original no llega a MIN_COMPARABLES.

    bedrooms entra como filtro exacto (no ±1, a diferencia de accommodates)
    porque, a igual capacidad, el número de habitaciones por sí solo ya
    explica una diferencia de precio real y grande entre anuncios (datos de
    Barcelona: 234€/mediana con 1 hab. frente a 297€ con 3 hab., para la
    misma capacidad 3-5) — mezclarlos en un mismo grupo inflaba la dispersión
    de la mediana sin motivo. Reduce el tamaño típico del conjunto (~20%) y
    sube algo la tasa de "alcance ampliado"/"insuficiente", un precio que
    merece la pena por un peer-group más comparable de verdad.

    anuncio_usuario necesita: room_type, accommodates, bedrooms,
    minimum_nights, neighbourhood, district. df_zona necesita: room_type,
    accommodates, bedrooms, minimum_nights, neighbourhood_cleansed y, si la
    ciudad tiene nivel de distrito propio, neighbourhood_group_cleansed (si
    no está la columna, "distrito" cae sobre la misma columna que "barrio",
    igual que ya hace find_similar_comparables en app.py para Málaga).
    """
    if alcance not in ALCANCES:
        raise ValueError(f"alcance debe ser uno de {ALCANCES}, recibido: {alcance!r}")

    tramo = tramo_estancia(anuncio_usuario["minimum_nights"])
    tiene_distrito = COL_DISTRITO in df_zona.columns

    base = df_zona[
        (df_zona["room_type"] == anuncio_usuario["room_type"])
        & df_zona["accommodates"].between(anuncio_usuario["accommodates"] - 1, anuncio_usuario["accommodates"] + 1)
        & (df_zona["bedrooms"] == anuncio_usuario["bedrooms"])
        & (_tramo_estancia_vectorizado(df_zona["minimum_nights"]) == tramo)
    ]

    def _en_alcance(alc: str) -> pd.DataFrame:
        if alc == "ciudad":
            return base
        if alc == "distrito" and tiene_distrito:
            return base[base[COL_DISTRITO] == anuncio_usuario["district"]]
        # "barrio", o "distrito" en una ciudad sin nivel de distrito propio.
        return base[base[COL_BARRIO] == anuncio_usuario["neighbourhood"]]

    n_en_alcance_original = len(_en_alcance(alcance))

    idx_inicio = ALCANCES.index(alcance)
    alcance_final = alcance
    resultado = _en_alcance(alcance)
    for alc in ALCANCES[idx_inicio:]:
        resultado = _en_alcance(alc)
        alcance_final = alc
        if len(resultado) >= MIN_COMPARABLES:
            break

    return ResultadoComparables(
        df=resultado,
        alcance_solicitado=alcance,
        alcance_final=alcance_final,
        se_amplio=(alcance_final != alcance),
        n_en_alcance_original=n_en_alcance_original,
        insuficiente=len(resultado) < MIN_COMPARABLES,
    )


# Pesos de cada eje en la distancia de proximidad. Capacidad y distancia
# geográfica pesan más (son lo primero que un anfitrión mira para decidir si
# dos anuncios "compiten" de verdad); baños pesa menos (con la misma
# capacidad y habitaciones, el número de baños distingue poco el precio).
# Suman 1.0.
PESOS_PROXIMIDAD = {
    "distancia": 0.30,
    "capacidad": 0.30,
    "habitaciones": 0.15,
    "estancia_minima": 0.15,
    "banos": 0.10,
}

_RADIO_TIERRA_M = 6_371_000.0


def _distancia_haversine_m(lat1, lon1, lat2, lon2):
    """Distancia haversine en metros. Acepta escalares o pd.Series/np.array."""
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _RADIO_TIERRA_M * np.arcsin(np.sqrt(a))


def _normalizar(valores: pd.Series) -> pd.Series:
    """Divide por la desviación típica de `valores` (z-score sin restar la
    media: aquí valores son diferencias absolutas, ya positivas, y lo único
    que importa es la escala). Con desviación 0 (todas las filas iguales en
    ese eje) no hay nada que distinga a las filas: devuelve 0 en vez de
    dividir por cero."""
    desviacion = valores.std(ddof=0)
    if not desviacion or pd.isna(desviacion):
        return pd.Series(0.0, index=valores.index)
    return valores / desviacion


def ordenar_por_proximidad(anuncio_usuario: dict, df_comparables: pd.DataFrame) -> pd.DataFrame:
    """Reordena df_comparables (normalmente el .df de un ResultadoComparables)
    de más a menos parecido a anuncio_usuario, con como mucho
    MAX_POR_ANFITRION filas por host_id.

    anuncio_usuario necesita: latitude, longitude, accommodates, bedrooms,
    bathrooms, minimum_nights. df_comparables necesita esas mismas columnas
    más host_id (no la carga hoy load_comparables en app.py: hace falta
    añadirla ahí para conectar esto a la interfaz).

    Añade una columna "distancia_m" (metros al anuncio del usuario) al
    resultado; el resto de columnas de df_comparables se conservan tal cual.
    No excluye reseñas ni historial del anfitrión de la distancia: explican
    el precio, no el parecido a ojos de un anfitrión.
    """
    if "host_id" not in df_comparables.columns:
        raise ValueError(
            "df_comparables necesita la columna 'host_id' para aplicar el tope de "
            f"{MAX_POR_ANFITRION} anuncios por anfitrión."
        )
    if df_comparables.empty:
        return df_comparables.assign(distancia_m=pd.Series(dtype=float))

    df = df_comparables.copy()
    df["distancia_m"] = _distancia_haversine_m(
        anuncio_usuario["latitude"], anuncio_usuario["longitude"], df["latitude"], df["longitude"],
    )

    diferencias = {
        "distancia": df["distancia_m"],
        "capacidad": (df["accommodates"] - anuncio_usuario["accommodates"]).abs(),
        "habitaciones": (df["bedrooms"] - anuncio_usuario["bedrooms"]).abs(),
        "banos": (df["bathrooms"] - anuncio_usuario["bathrooms"]).abs(),
        "estancia_minima": (df["minimum_nights"] - anuncio_usuario["minimum_nights"]).abs(),
    }
    puntuacion = sum(PESOS_PROXIMIDAD[eje] * _normalizar(valores) for eje, valores in diferencias.items())

    # mergesort es estable: a igualdad de puntuación, se conserva el orden de
    # entrada en vez de uno arbitrario, para que el resultado sea determinista.
    df = df.assign(_puntuacion=puntuacion).sort_values("_puntuacion", kind="mergesort")

    vistos: dict = {}
    conservar = []
    for host_id, idx in zip(df["host_id"], df.index):
        n_visto = vistos.get(host_id, 0)
        if n_visto < MAX_POR_ANFITRION:
            conservar.append(idx)
            vistos[host_id] = n_visto + 1

    return df.loc[conservar].drop(columns="_puntuacion")
