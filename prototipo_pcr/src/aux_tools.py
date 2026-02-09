"""
Todas las funciones auxiliares empleadas en los distintos calculos
"""

import src.parametros as params
import polars as pl
import datetime as dt
import calendar
import unicodedata
import re


def get_fecha_nivel(columna_nivel: str, niveles: list[str], prefijo: str) -> pl.Expr:
    """
    Construye una expresión condicional que devuelve la columna correspondiente
    según el valor en `columna_nivel`, usando prefijos.
    Ej: si nivel = 'recibo', selecciona 'fecha_inicio_vigencia_recibo'.
    """
    # Inicializa con la primera condición
    expr = pl.when(pl.col(columna_nivel) == niveles[0]).then(
        pl.col(f"{prefijo}_{niveles[0]}")
    )

    # Agrega el resto con .otherwise(pl.when(...).then(...))
    for nivel in niveles[1:]:
        expr = expr.otherwise(
            pl.when(pl.col(columna_nivel) == nivel).then(pl.col(f"{prefijo}_{nivel}"))
        )

    # Finaliza con None si no se cumple ningún nivel
    return expr


# Funcion que cambia el formato de fecha a AAAAMM
def yyyymm(col: pl.Expr) -> pl.Expr:
    return col.dt.year() * 100 + col.dt.month()


# Me dice si una fecha es cierre de mes
def es_ultimo_dia_mes(fecha: dt.date) -> bool:
    ultimo_dia = calendar.monthrange(fecha.year, fecha.month)[1]
    return fecha.day == ultimo_dia


# Devulve el mes anterior como entero en formato YYYYMM
def mes_anterior(yyyymm: int) -> int:
    anio = yyyymm // 100
    mes = yyyymm % 100
    if mes == 1:
        return (anio - 1) * 100 + 12
    else:
        return anio * 100 + (mes - 1)


# Funcion que calcula la diferencia entre dos fechas en días
def calcular_dias_diferencia(
    fecha_fin: pl.Expr,
    fecha_inicio: pl.Expr,
    incluir_extremos: bool = True,
) -> pl.Expr:
    return (fecha_fin - fecha_inicio).dt.total_days() + int(incluir_extremos)


def alinear_esquemas(dataframes: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """
    Alinea los tipos de datos y esquema de varios dataframes para poder unirlos
    """
    # Obtener el conjunto de todas las columnas y sus tipos más amplios
    columnas_union = {}
    for df in dataframes:
        for nombre, dtype in zip(df.columns, df.dtypes):
            # Si la columna ya existe, elige el tipo más amplio (por ejemplo, Float64 > Int64)
            if nombre in columnas_union:
                actual = columnas_union[nombre]
                # Convertimos a Float64 si hay mezcla Int/Float
                if (actual, dtype) in [(pl.Int64, pl.Float64), (pl.Float64, pl.Int64)]:
                    columnas_union[nombre] = pl.Float64
            else:
                columnas_union[nombre] = dtype
    # Convertir cada DataFrame al esquema común
    dataframes_ajustados = []
    for df in dataframes:
        cols_faltantes = [col for col in columnas_union if col not in df.columns]
        # Agregar columnas faltantes como nulas
        for col in cols_faltantes:
            df = df.with_columns(pl.lit(None).cast(columnas_union[col]).alias(col))
        # Asegurar tipos correctos
        df = df.select(
            [pl.col(col).cast(columnas_union[col]) for col in columnas_union]
        )
        dataframes_ajustados.append(df)

    return dataframes_ajustados


def estandarizar_nombre_columna(nombre):
    # Quita tildes
    nombre = (
        unicodedata.normalize("NFD", nombre).encode("ascii", "ignore").decode("utf-8")
    )
    nombre = nombre.lower()
    nombre = re.sub(r"[ -]+", "_", nombre)
    # Eliminar cualquier otro carácter no alfanumérico o _
    nombre = re.sub(r"[^\w_]", "", nombre)

    return nombre


def estandarizar_columnas(df: pl.DataFrame) -> pl.DataFrame:
    columnas_nuevas = [estandarizar_nombre_columna(col) for col in df.columns]

    return df.rename(dict(zip(df.columns, columnas_nuevas)))


def agregar_cohorte_dinamico(df: pl.DataFrame) -> pl.DataFrame:
    """
    La cohorte depende del tipo de contrato,
    por lo cual debe usar columnas distintas que deben aparecer en el insumo
    """
    columnas = df.columns
    tiene_col_directo = "fecha_expedicion_poliza" in columnas
    tiene_col_rea = "fe_ini_vig_contrato_reaseguro" in columnas
    if tiene_col_directo and tiene_col_rea:
        cohorte_directo = pl.col("fecha_expedicion_poliza").dt.year()
        cohorte_rea = pl.col("fe_ini_vig_contrato_reaseguro").dt.year()
        expr = (
            pl.when(pl.col("tipo_contrato") == "directo")
            .then(cohorte_directo)
            .otherwise(cohorte_rea)
        )
    elif tiene_col_directo:
        expr = pl.col("fecha_expedicion_poliza").dt.year()
    elif tiene_col_rea:
        expr = pl.col("fe_ini_vig_contrato_reaseguro").dt.year()
    else:
        raise ValueError(
            "No existen columnas de fecha válidas para definir la cohorte."
        )

    return df.with_columns(expr.alias("cohorte"))


def etiquetar_transicion(df: pl.DataFrame) -> pl.DataFrame:
    es_transicion = pl.col("fecha_valoracion") == params.FECHA_TRANSICION
    return df.with_columns(
        pl.when(es_transicion).then(pl.lit("1")).otherwise(pl.lit("0")).alias("transicion")
    )


def agregar_meses_fin(
        fecha: pl.Expr, 
        meses: pl.Expr | int
    ) -> pl.Expr:
    # Si meses es un entero fijo lo convierte a str
    meses_expr = pl.lit(meses) if isinstance(meses, int) else meses
    
    return (
        fecha
        .dt.offset_by(meses_expr.cast(pl.Utf8) + "mo")
        .dt.month_end()
    )