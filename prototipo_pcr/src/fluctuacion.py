"""
Funciones para el cálculo de la fluctuación por tasa de cambio de los componentes de la reserva
se condiciona porque la metodología cambia para ifrs17 vs ifrs4
"""

import src.aux_tools as aux_tools
import src.cruces as cruces
import polars as pl


def calc_fluctuacion(
    data_devengo: pl.DataFrame, tasas_cambio: pl.DataFrame
) -> pl.DataFrame:
    # Filtra para alicar solo a moneda extranjera

    data_devengo_mext = data_devengo.filter(pl.col("moneda") != "COP")
    data_devengo_mloc = data_devengo.filter(~(pl.col("moneda") != "COP"))

    # Cambios en tasa de cambio para usar segun el tipo de contabilidad
    # incluir columnas de estos deltas en el output
    delta_tc_bautizo = pl.col("tasa_cambio_fecha_valoracion") - pl.col(
        "tasa_cambio_fecha_constitucion"
    )
    delta_tc_mensual = pl.col("tasa_cambio_fecha_valoracion") - pl.col(
        "tasa_cambio_fecha_valoracion_anterior"
    )

    # usa el cambio en tasa respecto a la fecha de bautizo cuando es el primer mes de devengo
    es_mes_inicio = aux_tools.yyyymm(pl.col("fecha_constitucion")) == aux_tools.yyyymm(
        pl.col("fecha_valoracion")
    )
    delta_tc = pl.when(es_mes_inicio).then(delta_tc_bautizo).otherwise(delta_tc_mensual)

    data_devengo_mext = (
        data_devengo_mext.pipe(
            cruces.cruzar_tasas_cambio, tasas_cambio, liquidacion=False
        ) 
        # Se debe incluir el efecto de la acreditacion de intereses en la fluctuacion constitucion
        .with_columns(
            ((pl.col("saldo") + pl.col("acreditacion_intereses").fill_nan(0.0)) * delta_tc).alias("fluctuacion_constitucion")
        )
        # ANTES: .with_columns((pl.col("saldo") * delta_tc).alias("fluctuacion_constitucion"))
        .with_columns(
            # El signo de la liberacion se invierte para reflejar el efecto economico
            (-1 * pl.col("valor_liberacion") * delta_tc).alias("fluctuacion_liberacion")
        )
    )
    cols_fluc = ["fluctuacion_constitucion", "fluctuacion_liberacion"]
    cols_tasas = [c for c in data_devengo_mext.columns if "tasa_cambio_" in c]
    # concatena con los no fluctuados y llena nulos
    data_devengo_fluc = (
        pl.concat([data_devengo_mloc, data_devengo_mext], how="diagonal")
        .with_columns([pl.col(col).fill_null(0.0) for col in cols_fluc])
        .with_columns([pl.col(col).fill_null(1.0) for col in cols_tasas])
    )

    return data_devengo_fluc
