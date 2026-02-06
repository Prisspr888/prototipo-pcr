import polars as pl
import src.aux_tools as aux_tools

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
            aux_tools.yyyymm(pl.col("fecha")).alias("anames")
        ])
        .select(["anames", "indice_ipc"])
    )

def procesar_curvas_tasas(
    df_tasas: pl.DataFrame, 
    max_nodos_requeridos: int = 122 # Default para cubrir el mes 121 de traslape
) -> pl.DataFrame:
    """
    Procesa las curvas y GARANTIZA que existan todos los nodos hasta max_nodos_requeridos.
    Si la curva input llega solo hasta el 120, extrapola la última tasa al 121 y 122.
    """
    
    # 1. Identificar todas las curvas únicas (cohortes) en el insumo
    fechas_curvas = df_tasas.select("fecha_curva").unique()
    
    # 2. Crear un esqueleto completo (Producto Cartesiano: Fechas x Nodos 1..122)
    # Esto asegura que no falte ningún nodo en ninguna curva
    nodos_esqueleto = pl.DataFrame({"nodo": range(1, max_nodos_requeridos + 1)})
    
    esqueleto_completo = (
        fechas_curvas.join(nodos_esqueleto, how="cross")
    )
    
    # 3. Cruzar el esqueleto con los datos reales
    df_proc = (
        esqueleto_completo
        .join(
            df_tasas.select(["fecha_curva", "nodo", "tasa"]), 
            on=["fecha_curva", "nodo"], 
            how="left"
        )
        .sort(["fecha_curva", "nodo"])
    )
    
    # 4. Extrapolación (Flat Forward)
    # Si falta la tasa en el nodo 121, usamos la del 120 (Forward Fill)
    df_proc = df_proc.with_columns(
        pl.col("tasa").forward_fill().over("fecha_curva")
    )
    
    # 5. Cálculo de Factores
    df_proc = (
        df_proc
        .with_columns([
            convertir_tasa_ea_am(pl.col("tasa")).alias("tasa_mensual_real")
        ])
        .with_columns([
            # Factor de acumulación real: (1 + r)^t
            (1 + pl.col("tasa_mensual_real"))
            .cum_prod()
            .over("fecha_curva")
            .alias("factor_capitalizacion")
        ])
        .with_columns([
            # Factor de Descuento Puntual v_t
            (1 / pl.col("factor_capitalizacion")).alias("factor_desc_real")
        ])
        .with_columns([
            # Suma Prefija (Prefix Sum)
            pl.col("factor_desc_real")
            .cum_sum()
            .over("fecha_curva")
            .alias("sum_desc_real")
        ])
        .select([
            pl.col("fecha_curva"), 
            pl.col("nodo"),        
            pl.col("tasa_mensual_real"),
            pl.col("factor_capitalizacion"), 
            pl.col("factor_desc_real"),      
            pl.col("sum_desc_real")          
        ])
    )
    
    return df_proc

def generar_insumo_actuarial(
    df_inflacion: pl.DataFrame,
    df_tasas: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Función Orquestadora.
    """
    # Procesar IPC
    df_ipc_lookup = procesar_inflacion(df_inflacion)
    
    # Procesar Curvas con Extensión Automática a 122 meses
    df_curvas_lookup = procesar_curvas_tasas(df_tasas, max_nodos_requeridos=122)
    
    # Generar llave de cruce eficiente
    df_curvas_lookup = df_curvas_lookup.with_columns(
        aux_tools.yyyymm(pl.col("fecha_curva")).alias("anames_curva")
    )
    
    return df_ipc_lookup, df_curvas_lookup