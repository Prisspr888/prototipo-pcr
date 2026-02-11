"""
Módulo de preprocesamiento de datos
toma las tablas de parametros y datos de entrada y prepara los insumos para el devengamiento
cada tipo de insumo tiene una función para hacer explícitos los pasos específicos por tipo
"""

import polars as pl
import datetime as dt
import duckdb
import src.cruces as cruces
import src.parametros as params
import src.aux_tools as aux_tools

# la fecha de inicio y fin de vigencia dependiendo del nivel de detalle
fe_ini_vig_nivel = aux_tools.get_fecha_nivel(
    "nivel_detalle", params.NIVELES_DETALLE, "fecha_inicio_vigencia"
)
fe_fin_vig_nivel = aux_tools.get_fecha_nivel(
    "nivel_detalle", params.NIVELES_DETALLE, "fecha_fin_vigencia"
)

def prep_input_prima_directo(
    produccion_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    # realiza los cruces base entre produccion del directo y parametros
    input_prima_directo = (
        produccion_df.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
    )
    # la fecha de inicio vigencia se guarda en una columna segun el nivel de detalle
    input_prima_directo = input_prima_directo.with_columns(
        fe_ini_vig_nivel.alias("fecha_inicio_vigencia")
    )
    # la fecha de constitucion de reserva es la de emision del recibo
    # NOTA: La fecha de emision es la fecha de contabilizacion en SAP
    input_prima_directo = input_prima_directo.with_columns(
        pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
    )
    # la fecha de inicio y fin de vigencia dependiendo del nivel de detalle
    input_prima_directo = input_prima_directo.with_columns(
        pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
            "fecha_inicio_devengo"
        )
    )
    # la fecha de fin del devengo
    input_prima_directo = input_prima_directo.with_columns(
        fe_fin_vig_nivel.alias("fecha_fin_devengo")
    )
    # el valor base del devengo
    input_prima_directo = input_prima_directo.with_columns(
        pl.col("valor_prima_emitida").alias("valor_base_devengo")
    )
    return input_prima_directo


# Prepara el insumo de descuento comercial directo
def prep_input_dcto_directo(
    produccion_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    descuento_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    # se reconstruye la prima antes de otorgar los descuentos o prima tarifa
    pct_dcto_total = pl.col("podto_comercial") + pl.col("podto_tecnico")
    prima_reconstruida = pl.col("valor_prima_emitida") / (1 - pct_dcto_total)

    # cambia el tipo de insumo al descuento y aplica los cruces de parametros
    input_dcto_directo = (
        produccion_df.with_columns(pl.lit("dcto_directo").alias("tipo_insumo"))
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .pipe(cruces.cruzar_descuento, descuento_df)
    )
    # la fecha de constitucion de reserva es la de emision del recibo
    input_dcto_directo = (
        input_dcto_directo.filter(
            pl.col("fecha_contabilizacion_recibo") <= fe_valoracion
        )
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
        .with_columns(
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns(
            # valor base del devengo es el porc descuento comercial aplicado a la prima tarifa reconstruida
            (prima_reconstruida * pl.col("podto_comercial")).alias("valor_base_devengo")
        )
    )
    return input_dcto_directo


# Prepara el insumo de gasto de expedicion directo
def prep_input_gasto_directo(
    produccion_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    gasto_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    mapping_gastos = {
        "expedicion_comisiones": "gasto_comi_directo",
        "expedicion_otros": "gasto_otro_directo",
        "terremoto_factor_pprr": "terremoto_factor_pprr",
        "terremoto_nota_tecnica": "terremoto_nota_tecnica",
    }

    input_gasto_directo = (
        produccion_df.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .pipe(cruces.cruzar_gastos_expedicion, gasto_df)
        .with_columns(
            # mapea el tipo insumo segun el tipo de gasto
            tipo_insumo=pl.when(pl.col("ramo_sura").is_in(params.RAMOS_MATEMATICA))
            .then(
                pl.lit("matematica_local_")
                + pl.col("tipo_gasto").replace(mapping_gastos)
            )
            .otherwise(pl.col("tipo_gasto").replace(mapping_gastos))
        )
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
        .with_columns(
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns(
            # valor base del devengo es el porc de gasto aplicado sobre la prima emitida
            (pl.col("valor_prima_emitida") * pl.col("porc_gasto")).alias(
                "valor_base_devengo"
            )
        )
    )
    return input_gasto_directo


# Prepara el insumo de prima cedida reaseguro proporcional
def prep_input_prima_rea(
    cesion_rea_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    # realiza los cruces base entre cesion del reaseguro y parametros
    input_prima_rea = (
        cesion_rea_df.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            # se constituye en la fe emision recibo de rea
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
        .with_columns(
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns((pl.col("valor_prima_cedida")).alias("valor_base_devengo"))
    )
    return input_prima_rea


# Prepara el insumo de descuento comercial aplicado al reaseguro proporcional
def prep_input_dcto_rea(
    cesion_rea_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    descuento_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    # se reconstruye la prima antes de otorgar los descuentos o prima tarifa
    pct_dcto_total = pl.col("podto_comercial") + pl.col("podto_tecnico")
    prima_reconstruida_ced = pl.col("valor_prima_cedida") / (1 - pct_dcto_total)

    # cambia el tipo de insumo al descuento y aplica los cruces de parametros
    input_dcto_rea = (
        cesion_rea_df.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .with_columns(pl.lit("dcto_rea_prop").alias("tipo_insumo"))
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .pipe(cruces.cruzar_descuento, descuento_df, True)
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
        .with_columns(
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns(
            # valor base del devengo es el porc descuento comercial aplicado a la prima reconstruida cedida
            (prima_reconstruida_ced * pl.col("podto_comercial")).alias(
                "valor_base_devengo"
            )
        )
    )
    return input_dcto_rea


# Prepara el insumo de gasto expedicion del reaseguro proporcional
def prep_input_gasto_rea(
    cesion_rea_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    gasto_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    mapping_gastos = {
        "expedicion_comisiones": "gasto_comi_rea_prop",
        "expedicion_otros": "gasto_otro_rea_prop",
    }

    input_gasto_rea = (
        cesion_rea_df.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .pipe(cruces.cruzar_gastos_expedicion, gasto_df, True)
        .filter(pl.col("tipo_gasto").is_in(list(mapping_gastos.keys())))
        .with_columns(
            # mapea el tipo insumo segun el tipo de gasto
            tipo_insumo=pl.when(pl.col("ramo_sura").is_in(params.RAMOS_MATEMATICA))
            .then(
                pl.lit("matematica_local_")
                + pl.col("tipo_gasto").replace(mapping_gastos)
            )
            .otherwise(pl.col("tipo_gasto").replace(mapping_gastos))
        )
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
        .with_columns(
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns(
            # valor base del devengo es el porc de gasto aplicado sobre la prima cedida
            (pl.col("valor_prima_cedida") * pl.col("porc_gasto")).alias(
                "valor_base_devengo"
            )
        )
    )
    return input_gasto_rea


# Prepara el insumo de comision de reaseguro proporcional
def prep_input_comi_rea(
    comision_rea_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    # realiza los cruces base entre comision del reaseguro y parametros
    input_comi_rea = (
        comision_rea_df.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            # el pago de la comision debe tener su propio recibo
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
        .with_columns(
            # tal vez aqui es muy dificil mapear coberturas, si se pudiera usar la fecha de la cobertura
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns(pl.col("valor_comision_rea").alias("valor_base_devengo"))
    )
    return input_comi_rea


# Prepara el insumo de onerosidad
def prep_input_onerosidad(
    onerosidad_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    # realiza los cruces base entre registros de onerosidad y parametros
    input_onerosidad = (
        onerosidad_df.filter(pl.col("fecha_operacion") <= fe_valoracion)
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .with_columns(
            fecha_inicio_vigencia_recibo=pl.col("fecha_operacion"),
            fecha_fin_vigencia_recibo=pl.col("fecha_fin_vigencia_poliza"),
            fecha_inicio_vigencia_cobertura=pl.col("fecha_operacion"),
            fecha_fin_vigencia_cobertura=pl.col("fecha_fin_vigencia_poliza"),
        )
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            # la fecha de constitucion de reserva es la de la operacion que
            # afecta la onerosidad
            pl.col("fecha_operacion").alias("fecha_constitucion")
        )
        .with_columns(
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns(pl.col("valor_onerosidad").alias("valor_base_devengo"))
    )

    return input_onerosidad


def prep_input_recup_onerosidad_pp(
    onerosidad_df: pl.DataFrame,
    cesion_rea_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    # realiza los cruces base entre registros de onerosidad y parametros
    polizas_rea = cesion_rea_df.select(
        [
            "tipo_negocio",
            "contrato_reaseguro",
            "nit_reasegurador",
            "tipo_reasegurador",
            "porc_participacion_reasegurador",
            "fe_ini_vig_contrato_reaseguro",
            "fe_fin_vig_contrato_reaseguro",
            "poliza",
            "compania",
            "ramo_sura",
            "porc_cesion",
        ]
    ).unique()

    return (
        onerosidad_df.with_columns(tipo_insumo=pl.lit("recup_onerosidad_pp"))
        .filter(pl.col("fecha_operacion") <= fe_valoracion)
        .drop("tipo_negocio")
        .join(polizas_rea, on=["poliza", "compania", "ramo_sura"], how="inner")
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .with_columns(
            fecha_inicio_vigencia_recibo=pl.col("fecha_operacion"),
            fecha_fin_vigencia_recibo=pl.col("fecha_fin_vigencia_poliza"),
            fecha_inicio_vigencia_cobertura=pl.col("fecha_operacion"),
            fecha_fin_vigencia_cobertura=pl.col("fecha_fin_vigencia_poliza"),
        )
        .with_columns(fe_ini_vig_nivel.alias("fecha_inicio_vigencia"))
        .with_columns(
            # la fecha de constitucion de reserva es la de la operacion que
            # afecta la onerosidad
            pl.col("fecha_operacion").alias("fecha_constitucion")
        )
        .with_columns(
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_ini_vig_nivel]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(fe_fin_vig_nivel.alias("fecha_fin_devengo"))
        .with_columns(
            valor_base_devengo=pl.col("valor_onerosidad")
            * pl.col("porc_cesion")
            * pl.col("porc_participacion_reasegurador")
        )
    )


def cruzar_costo_seguim(
    costo_contrato: pl.DataFrame,
    seguimiento_costo: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    """
    Cruza los datos del pago de costo de contrato con el seguimiento mensual del contrato
    para la fecha de valoración de interés
    """
    fe_valoracion_str = fe_valoracion.strftime("%Y-%m-%d")
    return duckdb.sql(
        f"""
        SELECT 
            costo.*,
            seguim.limite_agregado_valor_instalado,
            seguim.valor_siniestros_incurridos_mes,
            seguim.valor_salvamentos_mes,
            seguim.limite_agregado_casos_instalado,
            seguim.casos_incurridos_mes
        FROM costo_contrato AS costo
            LEFT JOIN seguimiento_costo AS seguim
            ON costo.contrato_reaseguro == seguim.contrato_reaseguro
            AND seguim.fecha_cierre = DATE '{fe_valoracion_str}'
        """
    ).pl()


# Prepara el insumo de costo contrato reaseguro no proporcional
def prep_input_costo_con(
    costo_contrato: pl.DataFrame,
    seguimiento_costo: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    """
    El insumo del costo de contrato se puede devengar de dos formas diferentes,
    por la regla diaria o por el consumo del limite agregado del contrato.
    En cada periodo, se usa el que resulte en una liberación porcentual mayor.
    Por esto, la preparación del insumo requiere algunos pasos adicionales.
    """

    # si ya viene devengandose, se recalcula el devengamiento diario
    if "saldo_anterior" in costo_contrato.columns:
        base_devengo = pl.col(
            "saldo_anterior"
        )  # si es de un output anterior, ya viene abierto por reasegurador
        fe_inicio_devengo = pl.lit(
            fe_valoracion
        ).dt.month_start()  # recalcula el devengo desde esta fecha
    # si se procesa por primera vez el recibo, parte de la base inicial
    else:
        # si el costo de contrato viene totalizado en el recibo, se abre por reasegurador
        base_devengo = pl.col("valor_costo_contrato") + pl.col("valor_reinstalamento")
        fe_inicio_devengo = fe_ini_vig_nivel

    # realiza los cruces base entre cesion del reaseguro y parametros
    input_costo_con = (
        costo_contrato.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .pipe(
            # cruza con los datos de seguimiento del contrato
            cruzar_costo_seguim,
            seguimiento_costo,
            fe_valoracion,
        )
        .with_columns(
            # si hay que recalcular el devengo diario, usa un nuevo inicio de vigencia
            fe_inicio_devengo.alias("fecha_inicio_vigencia")
        )
        .with_columns(
            # se constituye siempre en la fecha del recibo del costo contrato
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
        .with_columns(
            # se inicia a devengar en la fecha max entre constitucion, inicio cobertura rea o fecha de reinstalamento
            pl.max_horizontal(
                [
                    pl.col("fecha_constitucion"),
                    pl.max_horizontal(
                        [
                            pl.coalesce(
                                [pl.col("fe_reinstalamento"), fe_inicio_devengo]
                            ),
                            fe_inicio_devengo,
                        ]
                    ),
                ]
            ).alias("fecha_inicio_devengo")
        )
        .with_columns(
            # el fin del devengo siempre es el fin de vigencia que cubre el recibo
            fe_fin_vig_nivel.alias("fecha_fin_devengo")
        )
        .with_columns(
            # los calculos deben estar abiertos por reasegurador (se garantiza en la condicion de arriba)
            (base_devengo).alias("valor_base_devengo")
        )
    )

    return input_costo_con


def prep_input_recup_onerosidad_np(
    recup_onerosidad_df: pl.DataFrame,
    seguimiento_costo: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    excepciones_df: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    """
    Asi como el insumo del costo de contrato, la recuperacion de onerosidad
    por contratos no proporcionales se puede devengar de dos formas diferentes,
    por la regla diaria o por el consumo del limite agregado del contrato.
    En cada periodo, se usa el que resulte en una liberación porcentual mayor.
    Por esto, la preparación del insumo requiere algunos pasos adicionales.
    """

    # si ya viene devengandose, se recalcula el devengamiento diario
    if "saldo_anterior" in recup_onerosidad_df.columns:
        base_devengo = pl.col(
            "saldo_anterior"
        )  # si es de un output anterior, ya viene abierto por reasegurador
        fe_inicio_devengo = pl.lit(
            fe_valoracion
        ).dt.month_start()  # recalcula el devengo desde esta fecha
    # si se procesa por primera vez el recibo, parte de la base inicial
    else:
        base_devengo = pl.col("valor_recuperacion") * pl.col(
            "porc_participacion_reasegurador"
        )
        fe_inicio_devengo = fe_ini_vig_nivel

    return (
        recup_onerosidad_df.with_columns(
            fecha_inicio_vigencia_recibo=pl.col("fe_ini_vig_contrato_reaseguro"),
            fecha_fin_vigencia_recibo=pl.col("fe_fin_vig_contrato_reaseguro"),
            fecha_inicio_vigencia_cobertura=pl.col("fe_ini_vig_contrato_reaseguro"),
            fecha_fin_vigencia_cobertura=pl.col("fe_fin_vig_contrato_reaseguro"),
        )
        .filter(pl.col("fecha_calculo_recuperacion") <= fe_valoracion)
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .pipe(cruces.cruzar_excepciones_50_50, excepciones_df)
        .pipe(
            # cruza con los datos de seguimiento del contrato
            cruzar_costo_seguim,
            seguimiento_costo,
            fe_valoracion,
        )
        .with_columns(
            # si hay que recalcular el devengo diario, usa un nuevo inicio de vigencia
            fe_inicio_devengo.alias("fecha_inicio_vigencia")
        )
        .with_columns(
            # se constituye siempre en la fecha del calculo de la recuperacion
            pl.col("fecha_calculo_recuperacion").alias("fecha_constitucion")
        )
        .with_columns(
            # se inicia a devengar en la fecha max entre constitucion o inicio cobertura rea
            pl.max_horizontal([pl.col("fecha_constitucion"), fe_inicio_devengo]).alias(
                "fecha_inicio_devengo"
            )
        )
        .with_columns(
            # el fin del devengo siempre es el fin de vigencia que cubre el recibo
            fe_fin_vig_nivel.alias("fecha_fin_devengo")
        )
        .with_columns(
            # los calculos deben estar abiertos por reasegurador (se garantiza en la condicion de arriba)
            (base_devengo).alias("valor_base_devengo")
        )
    )


def prep_input_cartera(
    cartera_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    return (
        cartera_df.pipe(cruces.cruzar_param_contabilidad, param_contabilidad)
        .with_columns(
            fecha_valoracion=fe_valoracion,
            fecha_inicio_periodo=pl.lit(fe_valoracion)
            .dt.month_start()
            .alias("fecha_inicio_periodo"),
            valor_md=pl.col("valor_ml"),
            anio_liberacion=pl.lit("no_aplica"),
            regla_devengo=pl.lit("no_aplica"),
        )
        .filter(
            pl.col("fecha_corte").is_between(
                pl.col("fecha_inicio_periodo"), pl.col("fecha_valoracion")
            )
        )
        .pipe(aux_tools.agregar_cohorte_dinamico)
        .pipe(aux_tools.etiquetar_transicion)
    )


def prep_input_produccion_arl(produccion_arl: pl.DataFrame) -> pl.DataFrame:
    return produccion_arl.with_columns(
        fecha_inicio_vigencia_recibo=pl.col("mes_cotizacion").dt.month_start(),
        fecha_inicio_vigencia_cobertura=pl.col("mes_cotizacion").dt.month_start(),
        fecha_fin_vigencia_recibo=pl.col("mes_cotizacion").dt.month_end(),
        fecha_fin_vigencia_cobertura=pl.col("mes_cotizacion").dt.month_end(),
        fecha_expedicion_poliza=pl.col("mes_cotizacion").dt.month_start(),
    )


def prep_input_componente_inversion(
    componente_inversion_df: pl.DataFrame,
    param_contabilidad: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    
    # realiza los cruces base entre produccion del directo y parametros
    input_componente_inversion_directo = (
        componente_inversion_df.filter(pl.col("fecha_contabilizacion_recibo") <= fe_valoracion)
        .pipe(cruces.cruzar_param_contabilidad, param_contabilidad)

        # la fecha de constitucion de reserva es la de emision del recibo
        # NOTA: La fecha de emision es la fecha de contabilizacion en SAP
        .with_columns(
            pl.col("fecha_contabilizacion_recibo").alias("fecha_constitucion")
        )
    )

    # Construir fe_fin_devengo
    fe_fin_devengo = (
        pl.when(pl.col("fecha_cancelacion_poliza").is_null())
        .then(
            pl.max_horizontal([
                pl.col("fecha_fin_vigencia_poliza"),
                pl.col("fecha_constitucion")
            ])
        )
        .otherwise(
            # Calcular el maximo 
            pl.max_horizontal([
                # Calcular el mínimo por grupo compania, ramo, poliza
                pl.min_horizontal(
                    [   
                        pl.when(pl.col("tipo_op").is_in(['16', '26']))
                        .then(pl.col("fecha_constitucion"))
                        .min()
                        .over(["compania", "ramo_sura", "poliza"]),

                        pl.col('fecha_fin_vigencia_poliza')
                    ]
                ),
                pl.col("fecha_cancelacion_poliza"),
                pl.col("fecha_constitucion")
            ])
        )
    )
    
    
    input_componente_inversion_directo = input_componente_inversion_directo.with_columns(
        # la fecha de inicio de devengo que depende de la vigencia de la póliza o la fecha
        # de contabilización del registro
        pl.max_horizontal([
            pl.col("fecha_constitucion"),
            pl.col('fecha_inicio_vigencia_poliza')
        ]).alias("fecha_inicio_devengo"),

        # la fecha de fin del devengo
        fe_fin_devengo.alias("fecha_fin_devengo"),
        
        # el valor base del devengo
        pl.col("valor_prima_emitida").alias("valor_base_devengo")
    )
    
    return input_componente_inversion_directo


def anexar_info_financiacion(
    input_prep: pl.DataFrame,
    param_financiacion: pl.DataFrame,
    indice_ipc: pl.DataFrame,
    factores_interes: pl.DataFrame,
    fe_valoracion: dt.date,
) -> pl.DataFrame:
    

    # Pre-definimos la lista de expresiones de cálculo cuando aplica financiacion
    columnas_calculadas = [
        aux_tools.yyyymm(pl.col('fecha_inicio_vigencia')).alias('mes_inicio_vigencia'),
        aux_tools.yyyymm(pl.col('fecha_fin_devengo')).alias('mes_fin_vigencia'),
        aux_tools.yyyymm(pl.lit(fe_valoracion)).alias('mes_valoracion'),
        aux_tools.yyyymm(pl.lit(fe_valoracion).dt.offset_by("-1mo")).alias('mes_valoracion_anterior'),
        (pl.col('fecha_inicio_vigencia').dt.month_end().dt.day() - pl.col('fecha_inicio_vigencia').dt.day() + 1).alias('dias_vig_ini'),
        pl.col('fecha_inicio_vigencia').dt.month_end().dt.day().alias('dias_nodo_ini'),
        pl.col('fecha_fin_devengo').dt.day().alias('dias_vig_fin'),
        pl.col('fecha_fin_devengo').dt.month_end().dt.day().alias('dias_nodo_fin')
    ]

    input_final = ( 
        input_prep
        # Cruza la tabla parametrica para identificar registros donde aplica
        .pipe(cruces.cruzar_parm_financiacion, param_financiacion)
        # Aplicamos el calculo solo donde aplica_comp_financ == 1
        .with_columns([
            pl.when(pl.col("aplica_comp_financ") == 1)
            .then(expr)
            .otherwise(pl.lit(None))
            .alias(expr.meta.output_name()) 
            for expr in columnas_calculadas
        ])
        # cruza los factores de la curva de interes e ipc
        .pipe(cruces.cruzar_factores_lir, indice_ipc, factores_interes)
    )

    return input_final