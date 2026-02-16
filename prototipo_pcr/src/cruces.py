import polars as pl
import duckdb


# Cruza tipo de insumo con todos los parametros contabilidad que le aplican
def cruzar_param_contabilidad(
    base: pl.DataFrame, param_contabilidad: pl.DataFrame
) -> pl.DataFrame:
    # Si tipo cont viene en base, se eliminan combinaciones invalidas y se debe usar es la de param_cont
    if "tipo_contabilidad" in base.columns:
        join_kind = (
            "INNER"  # usa un inner join para conservar solo combinaciones válidas
        )
        tipocont_key = "AND b.tipo_contabilidad = param_cont.tipo_contabilidad"
        exclude = (
            "EXCLUDE (tipo_contabilidad)"  # excluye tipo_contabilidad que trae la base
        )
    else:
        join_kind, tipocont_key, exclude = "LEFT", "", ""
    return duckdb.sql(
        f"""
        SELECT
             b.* {exclude}
            ,param_cont.tipo_contabilidad as tipo_contabilidad
            ,param_cont.componente
            ,param_cont.clasificacion_adicional
            ,param_cont.tipo_contrato
            ,param_cont.nivel_detalle
            ,param_cont.signo_constitucion
        FROM base AS b
            {join_kind} JOIN param_contabilidad AS param_cont
                ON b.tipo_insumo = param_cont.tipo_insumo
                AND b.tipo_negocio = param_cont.tipo_negocio
                {tipocont_key}
        """
    ).pl()


# Cruza porcentaje descuento con produccion
# las columnas del cruce pueden cambiar segun como venga la tabla del descuento
def cruzar_descuento(
    produccion: pl.DataFrame, descuento: pl.DataFrame, reaseguro=False
) -> pl.DataFrame:
    # usa sufijo de rea para que cruce el dscto con el recibo de rea y no del directo
    suffix_rea = "_rea" if reaseguro else ""
    return duckdb.sql(
        f"""
        SELECT
            prod.*
            , COALESCE(
                dcto.podto_comercial,
                0
            ) AS podto_comercial
            , COALESCE(
                dcto.podto_tecnico,
                0
            ) AS podto_tecnico
        FROM produccion AS prod
            LEFT JOIN descuento AS dcto
                ON prod.compania = dcto.compania
                AND prod.ramo_sura = dcto.ramo_sura
                AND CAST(prod.poliza AS VARCHAR) = CAST(dcto.poliza AS VARCHAR)
                AND prod.recibo = dcto.recibo{suffix_rea}
                AND prod.poliza_certificado = dcto.poliza_certificado
                AND prod.amparo = dcto.amparo
                AND prod.tipo_op = dcto.tipo_op
                AND prod.producto = dcto.producto
                AND prod.numero_documento_sap = dcto.numero_documento_sap
        """
    ).pl()

# Cruza produccion y gastos segun el nivel de detalle encontrado en la tabla gasto
def cruzar_gastos_expedicion(
    produccion: pl.DataFrame, gastos: pl.DataFrame, reaseguro=False
) -> pl.DataFrame:
    """
    Cruza base de primas con los porcentajes de gasto correspondientes,
    como el cruce puede ser anterior al cruce con parametros, garantiza que
    en este cruce aparezcan todas las combinaciones de la prima con tipo_contabilidad.
    Evita duplicados usando prioridad de coincidencia para usar comodines correctamente
    """
    validacion = duckdb.sql("""
        WITH gastos_priorizados AS (
            SELECT
                g.*,
                CASE 
                    WHEN canal <> '*' AND producto <> '*' THEN 1
                    WHEN canal <> '*' AND producto = '*' THEN 2
                    WHEN canal = '*' AND producto = '*' THEN 3
                END AS prioridad_match
            FROM gastos g
        ),
        cruce AS (
            SELECT 
                prod.*,
                g.tipo_contabilidad,
                g.tipo_gasto,
                g.porc_gasto,
                g.prioridad_match,
                ROW_NUMBER() OVER (
                    PARTITION BY 
                        prod.tipo_op,
                        prod.tipo_insumo,
                        prod.poliza,
                        prod.poliza_certificado,
                        prod.recibo,
                        prod.amparo,
                        prod.cdsubgarantia,
                        prod.numero_documento_sap,
                        g.tipo_gasto,
                        g.tipo_contabilidad
                    ORDER BY g.prioridad_match
                ) AS rn
            FROM produccion AS prod
            JOIN gastos_priorizados AS g
                ON prod.compania = g.compania
                AND prod.ramo_sura = g.ramo_sura
                AND (prod.canal = g.canal OR g.canal = '*')
                AND (prod.producto = g.producto OR g.producto = '*')
                AND prod.fecha_expedicion_poliza BETWEEN g.fecha_inicio AND g.fecha_fin
        ),
        tamaños AS (
            SELECT 
                tipo_gasto,
                COUNT(*) AS rows_in_partition
            FROM cruce
            GROUP BY 
                tipo_op,
                tipo_insumo,
                poliza,
                poliza_certificado,
                recibo,
                amparo,
                cdsubgarantia,
                tipo_gasto,
                numero_documento_sap
        )
        SELECT 
            rows_in_partition,
            LIST(DISTINCT tipo_gasto) AS tipo_gasto_unicos
        FROM tamaños
        GROUP BY rows_in_partition
        ORDER BY rows_in_partition;
    """)
    #print(validacion)

    # El cruce debe ser por fecha expedición póliza y no por fecha de contabilización
    return duckdb.sql("""
        -- prioridad de cruce por comodin para evitar duplicados
        WITH gastos_priorizados AS (
            SELECT
                g.*,
                CASE 
                    WHEN canal <> '*' AND producto <> '*' THEN 1  
                    WHEN canal <> '*' AND producto = '*' THEN 2
                    WHEN canal = '*' AND producto = '*' THEN 3
                END AS prioridad_match
            FROM gastos g
        ),
        cruce AS (
            SELECT 
                prod.*,
                g.tipo_contabilidad,
                g.tipo_gasto,
                g.porc_gasto,
                g.prioridad_match,
                ROW_NUMBER() OVER (
                    PARTITION BY 
                        prod.numero_documento_sap,
                        prod.tipo_op,
                        prod.tipo_insumo,
                        prod.poliza,
                        prod.poliza_certificado,
                        prod.recibo,
                        prod.amparo,
                        prod.cdsubgarantia,
                        g.tipo_gasto,
                        g.tipo_contabilidad
                    ORDER BY g.prioridad_match
                ) AS rn
            FROM produccion AS prod
            JOIN gastos_priorizados g
                ON prod.compania = g.compania
                AND prod.ramo_sura = g.ramo_sura
                AND (prod.canal = g.canal OR g.canal = '*')
                AND (prod.producto = g.producto OR g.producto = '*')
                AND prod.fecha_expedicion_poliza BETWEEN g.fecha_inicio AND g.fecha_fin
        )
        SELECT * 
        FROM cruce
        WHERE rn = 1
    """).pl()


def cruzar_excepciones_50_50(
    base: pl.DataFrame, excepciones: pl.DataFrame
) -> pl.DataFrame:
    """
    Asigna la marca candidato_50_50 = 0 cuando sea una excepcion a la regla 50/50,
    ignorando columnas no presentes en `base`.
    """
    # Aplica la regla por defecto (marca con 1) si la columna no existe
    if "candidato_devengo_50_50" not in base.columns:
        base = base.with_columns(pl.lit(1).alias("candidato_devengo_50_50"))
    # Filtra columnas relevantes (las que existen en base)
    columnas_base = set(base.columns)
    exc_cols = [col for col in excepciones.columns if col != "candidato_devengo_50_50"]
    # Filtra excepciones: solo considera filas que no exigen columnas ausentes
    excepciones_filtradas = excepciones.filter(
        pl.all_horizontal(
            [((pl.col(col) == "*") | pl.lit(col in columnas_base)) for col in exc_cols]
        )
    )

    # Itera sobre excepciones válidas
    for row in excepciones_filtradas.iter_rows(named=True):
        condition = pl.lit(True)
        for col in exc_cols:
            if col not in columnas_base:
                continue  # ignora columnas ausentes

            if row[col] != "*":
                val = row[col]
                col_dtype = base.schema[col]
                if col_dtype in [pl.Int64, pl.Int32]:
                    val = int(val)
                elif col_dtype == pl.Utf8:
                    val = str(val)
                condition &= pl.col(col) == val

        base = base.with_columns(
            pl.when(condition)
            .then(row["candidato_devengo_50_50"])
            .otherwise(pl.col("candidato_devengo_50_50"))
            .alias("candidato_devengo_50_50")
        )

    return base


def cruzar_tasas_cambio(
    base: pl.DataFrame,
    tasas_cambio: pl.DataFrame,
    val_anterior: bool = True,
    bautizo: bool = True,
    liquidacion: bool = True,
) -> pl.DataFrame:
    # Crea la parte opcional del query para cruzar otras tasas
    tasa_bool = [val_anterior, bautizo, liquidacion]
    tasa_col = [
        ", COALESCE(tfval_ant.tasa_cambio, 1) AS tasa_cambio_fecha_valoracion_anterior",
        ", COALESCE(tconst.tasa_cambio, 1) AS tasa_cambio_fecha_constitucion",
        ", COALESCE(tliquid.tasa_cambio, 1) AS tasa_cambio_fecha_liquidacion",
    ]
    tasa_join = [
        """LEFT JOIN tasas_cambio AS tfval_ant
            ON base.fecha_valoracion_anterior = tfval_ant.fecha
            AND base.moneda = tfval_ant.moneda_origen""",
        """LEFT JOIN tasas_cambio AS tconst
            ON base.fecha_constitucion = tconst.fecha
            AND base.moneda = tconst.moneda_origen""",
        """LEFT JOIN tasas_cambio AS tliquid
           ON base.fecha_fin_devengo = tliquid.fecha
           AND base.moneda = tliquid.moneda_origen""",
    ]
    cols_query, joins_query = "", ""
    for tb, tc, tj in zip(tasa_bool, tasa_col, tasa_join):
        if tb:
            cols_query += "\n" + tc
            joins_query += "\n" + tj

    return duckdb.sql(
        f"""
        SELECT
            base.*
            , COALESCE(tfval.tasa_cambio, 1) AS tasa_cambio_fecha_valoracion
            {cols_query}
        FROM base
        LEFT JOIN tasas_cambio AS tfval
            ON base.fecha_valoracion = tfval.fecha
            AND base.moneda = tfval.moneda_origen
            {joins_query}
        """
    ).pl()


def cruzar_parm_financiacion(
    base: pl.DataFrame,
    param_compfinanc: pl.DataFrame
) -> pl.DataFrame:
    
    # Identificador temporal para asegurar integridad
    df_base_con_id = base.with_row_index("_temp_id")
    
    con = duckdb.connect()
    
    query = """
        WITH params_priorizados AS (
            SELECT 
                * ,
                ( (CASE WHEN tipo_insumo = '*' THEN 1 ELSE 0 END) +
                  (CASE WHEN compania = '*' THEN 1 ELSE 0 END) +
                  (CASE WHEN ramo_sura = '*' THEN 1 ELSE 0 END) +
                  (CASE WHEN producto = '*' THEN 1 ELSE 0 END) +
                  (CASE WHEN tipo_op = '*' THEN 1 ELSE 0 END)
                ) AS nivel_comodin
            FROM param_compfinanc
        ),
        cruce_con_prioridad AS (
            SELECT 
                b._temp_id,
                -- Se usa el nombre del CTE y el nombre de columna corregido
                COALESCE(params_priorizados.aplica_comp_financ, 0) as aplica_comp_financ,
                COALESCE(params_priorizados.aplica_ipc_mensual, 0) as aplica_ipc_mensual,
                params_priorizados.pais_curva,
                params_priorizados.moneda_curva,
                params_priorizados.meses_max_vigencia,
                ROW_NUMBER() OVER (
                    PARTITION BY b._temp_id 
                    ORDER BY params_priorizados.nivel_comodin ASC
                ) as rn
            FROM df_base_con_id AS b
            LEFT JOIN params_priorizados 
                ON (b.tipo_contabilidad = params_priorizados.tipo_contabilidad) -- debe especificarse
                AND (b.moneda = params_priorizados.moneda) -- debe especificarse
                AND (b.tipo_insumo = params_priorizados.tipo_insumo OR params_priorizados.tipo_insumo = '*')
                AND (b.compania = params_priorizados.compania OR params_priorizados.compania = '*')
                AND (b.ramo_sura = params_priorizados.ramo_sura OR params_priorizados.ramo_sura = '*')
                AND (b.producto = params_priorizados.producto OR params_priorizados.producto = '*')
                AND (b.tipo_op = params_priorizados.tipo_op OR params_priorizados.tipo_op = '*')
        )
        SELECT p_final.* EXCLUDE (rn, _temp_id)
        FROM (
            SELECT b.*, c.* EXCLUDE (_temp_id)
            FROM df_base_con_id b
            LEFT JOIN cruce_con_prioridad c ON b._temp_id = c._temp_id
            WHERE c.rn = 1
        ) AS p_final
        ORDER BY _temp_id 
    """
    
    return con.execute(query).pl()


def cruzar_factores_lir(
    base: pl.DataFrame,
    factor_ipc: pl.DataFrame,
    factores_interes: pl.DataFrame,
) -> pl.DataFrame:
    """
    Cruza la base de devengo con los factores financieros de ipc e interes bloqueado o 
    interes de nacimiento (lir) necesarios para el calculo de la reserva y sus componentes contables
    
    :param base: base de producción, gastos o elementos sujetos a reserva PCR
    :type base: pl.DataFrame
    :param factor_ipc: tabla de ipc generada en el módulo curvas_financiacion
    :type factor_ipc: pl.DataFrame
    :param factores_interes: tabla de factores de curvas de interes generada en curvas_fianciacion
    :type factores_interes: pl.DataFrame
    :return: base consolidada con los elementos necesarios para el calculo del saldo de reserva a tasa bloqueada - lir
    :rtype: DataFrame
    """
    con = duckdb.connect()
    
    return con.execute(
        f"""
            SELECT
                base.*,
                -- Lógica de IPC
                CASE 
                    WHEN base.aplica_ipc_mensual = 1 THEN ipclir.indice_ipc 
                    ELSE 1 
                END AS indice_ipc_ini,
                CASE 
                    WHEN base.aplica_ipc_mensual = 0 THEN ipclir.tasa 
                    ELSE 1 
                END AS tasa_ipc_ini,
                CASE 
                    WHEN base.aplica_ipc_mensual = 1 THEN ipcval.indice_ipc 
                    ELSE 1 
                END AS indice_ipc_actual,
                CASE 
                    WHEN base.aplica_ipc_mensual = 0 THEN ipcval.tasa
                    ELSE 1
                END AS tasa_ipc_actual,
                CASE 
                    WHEN base.aplica_ipc_mensual = 1 THEN ipcant.indice_ipc 
                    ELSE 1 
                END AS indice_ipc_anterior,
                CASE 
                    WHEN base.aplica_ipc_mensual = 0 THEN ipcant.tasa 
                    ELSE 1 
                END AS tasa_ipc_anterior,

                -- Factores de Interés (LIR)
                intereslir_val.factor_acumulacion AS fact_acum_val,
                intereslir_val.sum_desc_real AS sum_desc_lir_val,
                intereslir_val.tasa_fwd_real AS tasa_fwd_real_val,
                intereslir_ant.factor_acumulacion AS fact_acum_ant,
                intereslir_ant.sum_desc_real AS sum_desc_lir_ant,
                intereslir_ant.tasa_fwd_real AS tasa_fwd_real_ant,

                intereslir_ini.sum_desc_real AS desc_lir_nodo_ini,

                intereslir_fin.sum_desc_real AS sum_desc_lir_nodo_fin,
                intereslir_fin.factor_desc_real AS desc_lir_nodo_fin
                
            FROM base
            -- Cruces de IPC usando mesid_ipc
            LEFT JOIN factor_ipc AS ipclir
                ON base.mes_inicio_vigencia = ipclir.mesid_ipc
            LEFT JOIN factor_ipc AS ipcval
                ON base.mes_valoracion = ipcval.mesid_ipc
            LEFT JOIN factor_ipc AS ipcant
                ON base.mes_valoracion_anterior = ipcant.mesid_ipc
            
            -- Cruces de Interés usando mesid para curvas y valoración
            LEFT JOIN factores_interes AS intereslir_ini
                ON base.moneda_curva = intereslir_ini.moneda_curva
                AND base.pais_curva = intereslir_ini.pais_curva
                AND base.mes_inicio_vigencia = intereslir_ini.mesid_curva
                AND intereslir_ini.nodo = 1

            LEFT JOIN factores_interes AS intereslir_val
                ON base.moneda_curva = intereslir_val.moneda_curva
                AND base.pais_curva = intereslir_val.pais_curva
                AND base.mes_inicio_vigencia = intereslir_val.mesid_curva
                AND base.mes_valoracion = intereslir_val.mesid_valoracion
                
            LEFT JOIN factores_interes AS intereslir_ant
                ON base.moneda_curva = intereslir_ant.moneda_curva
                AND base.pais_curva = intereslir_ant.pais_curva
                AND base.mes_inicio_vigencia = intereslir_ant.mesid_curva
                AND base.mes_valoracion_anterior = intereslir_ant.mesid_valoracion

            LEFT JOIN factores_interes AS intereslir_fin
                ON base.moneda_curva = intereslir_fin.moneda_curva
                AND base.pais_curva = intereslir_fin.pais_curva
                AND base.mes_inicio_vigencia = intereslir_fin.mesid_curva
                AND base.mes_fin_vigencia = intereslir_fin.mesid_valoracion
        """
        ).pl()