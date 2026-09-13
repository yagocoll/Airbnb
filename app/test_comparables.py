"""Tests de comparables.py. Sin pytest en el entorno: unittest de la
librería estándar, se ejecuta con `python3 -m unittest test_comparables -v`.
"""

import unittest

import numpy as np
import pandas as pd

from comparables import (
    MAX_POR_ANFITRION,
    MIN_COMPARABLES,
    _normalizar,
    conjunto_comparables,
    ordenar_por_proximidad,
    tramo_estancia,
)

USUARIO_BASE = {
    "room_type": "Entire home/apt",
    "accommodates": 4,
    "bedrooms": 1,
    "bathrooms": 1.0,
    "minimum_nights": 2,
    "latitude": 41.3874,
    "longitude": 2.1686,
    "neighbourhood": "el Fort Pienc",
    "district": "Eixample",
}


def _fila(
    id_,
    host_id,
    lat=41.3874,
    lon=2.1686,
    room_type="Entire home/apt",
    accommodates=4,
    bedrooms=1,
    bathrooms=1.0,
    minimum_nights=2,
    neighbourhood="el Fort Pienc",
    district="Eixample",
):
    return {
        "id": id_,
        "host_id": host_id,
        "latitude": lat,
        "longitude": lon,
        "room_type": room_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "minimum_nights": minimum_nights,
        "neighbourhood_cleansed": neighbourhood,
        "neighbourhood_group_cleansed": district,
    }


class TestTramoEstancia(unittest.TestCase):
    def test_cajones(self):
        self.assertEqual(tramo_estancia(1), "corta")
        self.assertEqual(tramo_estancia(3), "corta")
        self.assertEqual(tramo_estancia(4), "media")
        self.assertEqual(tramo_estancia(7), "media")
        self.assertEqual(tramo_estancia(8), "larga")
        self.assertEqual(tramo_estancia(365), "larga")


class TestConjuntoComparables(unittest.TestCase):
    def test_filtro_duro_room_type_nunca_se_relaja(self):
        # 40 filas de habitación privada en el mismo barrio: aunque sobren
        # comparables, ninguna debe colarse si el usuario pide vivienda entera.
        filas = [_fila(i, host_id=i, room_type="Private room") for i in range(40)]
        df = pd.DataFrame(filas)
        resultado = conjunto_comparables(USUARIO_BASE, df, alcance="barrio")
        self.assertTrue(resultado.df.empty)
        self.assertTrue(resultado.insuficiente)

    def test_filtro_duro_bedrooms_exacto(self):
        # A diferencia de accommodates (±1), bedrooms es exacto: ni siquiera
        # una habitación de diferencia debe colarse.
        filas = [
            _fila(0, host_id=0, bedrooms=1),  # igual que USUARIO_BASE: entra
            _fila(1, host_id=1, bedrooms=2),  # una más: fuera
            _fila(2, host_id=2, bedrooms=0),  # una menos (estudio): fuera
        ]
        df = pd.DataFrame(filas)
        resultado = conjunto_comparables(USUARIO_BASE, df, alcance="ciudad")
        self.assertEqual(list(resultado.df["id"]), [0])

    def test_capacidad_mas_menos_uno(self):
        filas = [
            _fila(0, host_id=0, accommodates=3),  # dentro (±1)
            _fila(1, host_id=1, accommodates=4),  # dentro
            _fila(2, host_id=2, accommodates=5),  # dentro (±1)
            _fila(3, host_id=3, accommodates=6),  # fuera
            _fila(4, host_id=4, accommodates=2),  # fuera
        ]
        df = pd.DataFrame(filas)
        resultado = conjunto_comparables(USUARIO_BASE, df, alcance="ciudad")
        self.assertEqual(sorted(resultado.df["id"]), [0, 1, 2])

    def test_tramo_estancia_es_cajon_no_rango(self):
        # Usuario en tramo corto (2 noches). Un comparable a 3 noches es del
        # mismo cajón (corta) y debe entrar; uno a 8 noches es "larga" y no
        # debe colarse aunque esté numéricamente "cerca" en noches.
        filas = [
            _fila(0, host_id=0, minimum_nights=3),
            _fila(1, host_id=1, minimum_nights=8),
        ]
        df = pd.DataFrame(filas)
        resultado = conjunto_comparables(USUARIO_BASE, df, alcance="ciudad")
        self.assertEqual(list(resultado.df["id"]), [0])

    def test_respaldo_amplia_alcance_si_faltan_comparables(self):
        # Solo 5 en el barrio, pero 35 en el distrito (barrio distinto,
        # mismo distrito): debe ampliar de "barrio" a "distrito".
        filas = [_fila(i, host_id=i, neighbourhood="el Fort Pienc") for i in range(5)]
        filas += [
            _fila(i, host_id=i, neighbourhood="la Sagrada Família", district="Eixample")
            for i in range(5, 40)
        ]
        df = pd.DataFrame(filas)
        resultado = conjunto_comparables(USUARIO_BASE, df, alcance="barrio")
        self.assertTrue(resultado.se_amplio)
        self.assertEqual(resultado.alcance_final, "distrito")
        self.assertEqual(resultado.n_en_alcance_original, 5)
        self.assertEqual(len(resultado.df), 40)
        self.assertFalse(resultado.insuficiente)

    def test_respaldo_salta_con_29_no_con_30(self):
        filas_29 = [_fila(i, host_id=i) for i in range(29)]
        resultado_29 = conjunto_comparables(USUARIO_BASE, pd.DataFrame(filas_29), alcance="ciudad")
        self.assertTrue(resultado_29.insuficiente)

        filas_30 = [_fila(i, host_id=i) for i in range(30)]
        resultado_30 = conjunto_comparables(USUARIO_BASE, pd.DataFrame(filas_30), alcance="barrio")
        self.assertFalse(resultado_30.se_amplio)
        self.assertFalse(resultado_30.insuficiente)
        self.assertEqual(len(resultado_30.df), MIN_COMPARABLES)

    def test_insuficiente_incluso_a_nivel_ciudad(self):
        filas = [_fila(i, host_id=i) for i in range(10)]
        resultado = conjunto_comparables(USUARIO_BASE, pd.DataFrame(filas), alcance="barrio")
        self.assertEqual(resultado.alcance_final, "ciudad")
        self.assertTrue(resultado.insuficiente)

    def test_distrito_cae_sobre_barrio_sin_columna_de_distrito(self):
        # Málaga: sin neighbourhood_group_cleansed. "distrito" no debe fallar
        # ni comportarse como "ciudad" entera; debe caer sobre barrio.
        filas = [_fila(i, host_id=i, neighbourhood="el Fort Pienc") for i in range(35)]
        filas += [_fila(i, host_id=i, neighbourhood="otro barrio") for i in range(35, 70)]
        df = pd.DataFrame(filas).drop(columns="neighbourhood_group_cleansed")
        resultado = conjunto_comparables(USUARIO_BASE, df, alcance="distrito")
        self.assertEqual(len(resultado.df), 35)

    def test_alcance_invalido_lanza_error(self):
        with self.assertRaises(ValueError):
            conjunto_comparables(USUARIO_BASE, pd.DataFrame([_fila(0, 0)]), alcance="provincia")


class TestOrdenarPorProximidad(unittest.TestCase):
    def test_anuncio_identico_sale_primero_con_distancia_cero(self):
        filas = [
            _fila(0, host_id=0, lat=41.40, lon=2.20),
            _fila(1, host_id=1, lat=USUARIO_BASE["latitude"], lon=USUARIO_BASE["longitude"]),  # idéntico
            _fila(2, host_id=2, lat=41.35, lon=2.10),
        ]
        df = pd.DataFrame(filas)
        ordenado = ordenar_por_proximidad(USUARIO_BASE, df)
        self.assertEqual(ordenado.iloc[0]["id"], 1)
        self.assertAlmostEqual(ordenado.iloc[0]["distancia_m"], 0.0, places=6)

    def test_cambiar_capacidad_reordena(self):
        filas = [
            _fila(0, host_id=0, accommodates=4),  # igual que el usuario: más cerca
            _fila(1, host_id=1, accommodates=8),  # muy distinto: más lejos
        ]
        df = pd.DataFrame(filas)
        orden_1 = list(ordenar_por_proximidad(USUARIO_BASE, df)["id"])
        self.assertEqual(orden_1, [0, 1])

        # Si ahora la fila 1 tiene la misma capacidad que el usuario y la 0 se
        # aleja mucho en capacidad, el orden debe invertirse.
        filas2 = [
            _fila(0, host_id=0, accommodates=16),
            _fila(1, host_id=1, accommodates=4),
        ]
        df2 = pd.DataFrame(filas2)
        orden_2 = list(ordenar_por_proximidad(USUARIO_BASE, df2)["id"])
        self.assertEqual(orden_2, [1, 0])

    def test_maximo_dos_por_anfitrion(self):
        # 5 anuncios del mismo host a distintas distancias + 1 de otro host.
        filas = [_fila(i, host_id="host_A", lat=41.3874 + 0.001 * i, lon=2.1686) for i in range(5)]
        filas.append(_fila(5, host_id="host_B", lat=41.40, lon=2.20))
        df = pd.DataFrame(filas)
        ordenado = ordenar_por_proximidad(USUARIO_BASE, df)

        conteo_host_a = (ordenado["host_id"] == "host_A").sum()
        self.assertLessEqual(conteo_host_a, MAX_POR_ANFITRION)
        # Se quedan los 2 MÁS CERCANOS de host_A (ids 0 y 1, los de menor
        # distancia_m), no dos cualquiera.
        ids_host_a = set(ordenado.loc[ordenado["host_id"] == "host_A", "id"])
        self.assertEqual(ids_host_a, {0, 1})

    def test_falta_host_id_lanza_error_explicito(self):
        df = pd.DataFrame([_fila(0, host_id=0)]).drop(columns="host_id")
        with self.assertRaises(ValueError):
            ordenar_por_proximidad(USUARIO_BASE, df)

    def test_conjunto_vacio_no_falla(self):
        df = pd.DataFrame([_fila(0, host_id=0)]).iloc[0:0]
        ordenado = ordenar_por_proximidad(USUARIO_BASE, df)
        self.assertTrue(ordenado.empty)


class TestNormalizarPorDispersionDeZona(unittest.TestCase):
    def test_misma_distancia_normaliza_distinto_segun_dispersion_de_la_zona(self):
        distancia_m = 500.0
        # Zona "compacta" (Barcelona-like): comparables muy cerca entre sí ->
        # poca dispersión de distancias.
        distancias_compacta = pd.Series([distancia_m, 50.0, 80.0, 30.0, 60.0])
        # Zona "dispersa" (Mallorca/Euskadi-like): comparables a kilómetros
        # unos de otros -> mucha dispersión.
        distancias_dispersa = pd.Series([distancia_m, 5000.0, 8000.0, 3000.0, 6000.0])

        normalizado_compacta = _normalizar(distancias_compacta).iloc[0]
        normalizado_dispersa = _normalizar(distancias_dispersa).iloc[0]

        self.assertNotAlmostEqual(normalizado_compacta, normalizado_dispersa)
        # Los mismos 500 m "pesan" más (normalizan a un valor mayor) en una
        # zona compacta que en una dispersa.
        self.assertGreater(normalizado_compacta, normalizado_dispersa)

    def test_desviacion_cero_no_divide_por_cero(self):
        resultado = _normalizar(pd.Series([10.0, 10.0, 10.0]))
        self.assertTrue((resultado == 0.0).all())


if __name__ == "__main__":
    unittest.main()
