import polars as pl
import src.aux_tools as aux_tools
import src.parametros as params

def convertir_tasa_ea_am(tasa_ea: pl.Expr) -> pl.Expr:
    return (1 + tasa_ea).pow(1/12) - 1

def procesar_inflacion(df_inflacion: pl.DataFrame) -> pl.DataFrame:
    """
    Genera el vector de Índice de Precios Acumulado (IPC_t).
    """
    return (
        df_inflacion
        .sort("fecha")
        .with_columns([
            (1 + pl.col("tasa")).cum_prod().alias("indice_ipc"),
            aux_tools.yyyymm(pl.col("fecha")).alias("mesid_ipc")
        ])
        .select(["mesid_ipc", "indice_ipc"])
    )

def procesar_curvas_tasas(
    df_tasas: pl.DataFrame,
    df_param_compfin: pl.DataFrame
) -> pl.DataFrame:
    """
    Procesa las curvas garantizando que existan todos los nodos hasta el máximo definido en parámetros por moneda
    """
    # Prepara parámetros: obtiene el máximo de meses_max_vigencia por cada moneda/pais que aplica
    df_nodos_req = (
        df_param_compfin
        .filter(pl.col("aplica_comp_financ") == 1)
        .group_by(["pais_curva", "moneda_curva"])
        .agg(pl.col("meses_max_vigencia").max().alias("nodos_requeridos"))
    )

    # Filtra y une las tasas con sus requerimientos de nodos
    df_tasas = (
        df_tasas
        .join(
            df_nodos_req, 
            left_on=["pais", "moneda"], 
            right_on=["pais_curva", "moneda_curva"], 
            how="inner"
        )
        # Descarta nodos que exceden la vigencia máxima parametrizada + 2 por fechas intermedias
        .filter(pl.col("mes") <= pl.col("nodos_requeridos").cast(pl.Int64) + 2)
    )

    # Valida que existan los nodos necesarios comparando contra el parámetro de la tabla
    validacion = (
        df_tasas
        .group_by(["fecha_clave", "pais", "moneda", "nodos_requeridos"])
        .agg(
            pl.col("mes").max().alias("max_nodo_encontrado")
        )
    )

    # Identificar curvas incompletas respecto a su propio parámetro meses_max_vigencia
    curvas_incompletas = validacion.filter(
        pl.col("max_nodo_encontrado") < pl.col("nodos_requeridos").cast(pl.Int64) + 2
    )

    if curvas_incompletas.height > 0:
        detalle_error = curvas_incompletas.select(["fecha_clave", "pais", "moneda", "nodos_requeridos", "max_nodo_encontrado"])
        raise ValueError(
            f"ERROR INSUMOS INTERES REAL:\n"
            f"Curvas incompletas requieren nodos requeridos + 2 nodos:\n{detalle_error}."
        )

    # Cálculo de factores por ["fecha_clave", "pais", "moneda"]
    df_fact_financieros = (
        df_tasas
        .with_columns(
            pl.col("fecha_clave").cast(pl.Utf8).str.to_date("%Y%m%d")
        )
        .sort(["fecha_clave", "pais", "moneda", "mes"])
        .with_columns([
            convertir_tasa_ea_am(pl.col("tasa_interes")).alias("tasa_mensual_real")
        ])
        .with_columns([
            # Factor de acumulación real: a_t = (1 + r)^t
            (1 + pl.col("tasa_mensual_real"))
            .cum_prod()
            .over(["fecha_clave", "pais", "moneda"])
            .alias("factor_acumulacion")
        ])
        .with_columns([
            # Factor de descuento v_t
            (1 / pl.col("factor_acumulacion")).alias("factor_desc_real")
        ])
        .with_columns([
            # descuento real acumulado
            pl.col("factor_desc_real")
            .cum_sum()
            .over(["fecha_clave", "pais", "moneda"])
            .alias("sum_desc_real")
        ])
        .with_columns([
            pl.col("fecha_clave").alias("fecha_curva"),
            aux_tools.agregar_meses_fin(pl.col('fecha_clave'), pl.col('mes')).alias('fecha_valoracion')
        ])
        .with_columns([
            # llaves de cruce para devengo
            aux_tools.yyyymm(pl.col("fecha_curva")).alias("mesid_curva"),
            aux_tools.yyyymm(pl.col('fecha_valoracion')).alias('mesid_valoracion')
        ])
        .select([
            pl.col('mesid_curva'),
            pl.col("fecha_curva"),
            pl.col("moneda").alias("moneda_curva"),
            pl.col("pais").alias("pais_curva"),
            pl.col("mes").alias("nodo"),
            pl.col('mesid_valoracion'),
            pl.col('fecha_valoracion'),        
            pl.col("tasa_mensual_real"),
            pl.col("factor_acumulacion"), 
            pl.col("factor_desc_real"),      
            pl.col("sum_desc_real")          
        ])
    )
    
    return df_fact_financieros