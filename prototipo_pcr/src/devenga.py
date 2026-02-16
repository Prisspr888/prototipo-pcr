"""
Módulo de funciones de devengamiento de un componente durante una vigencia determinada
"""

import polars as pl
import datetime as dt
import src.parametros as params
import src.aux_tools as aux_tools


def deveng_diario(input_deveng: pl.DataFrame) -> pl.DataFrame:
    """
    Recibe un input preprocesado de devengamiento y devuelve el devengo uniforme diario
    """
    output_deveng_diario = (
        input_deveng.with_columns(
                pl.when(pl.col("fecha_constitucion") > pl.col("fecha_fin_devengo"))
                .then(pl.lit("entra_devengado"))
                .when(pl.col("fecha_valoracion") < pl.col("fecha_inicio_devengo"))
                .then(pl.lit("no_iniciado"))
                .when(
                        # si ya inicio a devengar o le quedan dias por devengar esta en curso
                        (pl.col("fecha_inicio_devengo") <= pl.col("fecha_valoracion"))
                        & 
                        (pl.col("fecha_valoracion") < pl.col("fecha_fin_devengo"))
                )
                .then(pl.lit("en_curso"))
                .otherwise(pl.lit("finalizado"))
                .alias("estado_devengo")
        )
        .with_columns(
            aux_tools.calcular_dias_diferencia(
                pl.col("fecha_fin_devengo"), pl.col("fecha_inicio_vigencia")
            ).alias("dias_constitucion")
        )
        .with_columns(
            # dias que ya se devengaron, depende del estado
            pl.when(pl.col("estado_devengo") == "no_iniciado")
            .then(pl.lit(0))
            .when(
                (pl.col("estado_devengo") == "en_curso")
                | (pl.col("estado_devengo") == "entra_devengado")
            )
            .then(
                aux_tools.calcular_dias_diferencia(
                    pl.min_horizontal(
                        [pl.col("fecha_fin_devengo"), pl.col("fecha_valoracion")]
                    ),
                    pl.col("fecha_inicio_vigencia"),
                )
            )
            .when(pl.col("estado_devengo") == "finalizado")
            .then(pl.col("dias_constitucion"))
            .alias("dias_devengados")
        )
        .with_columns(
            # dias aun no devengados segun el estado
            pl.when(pl.col("estado_devengo") == "no_iniciado")
            .then(pl.col("dias_constitucion"))
            .when(pl.col("estado_devengo") == "en_curso")
            .then(
                # no se incluye extremo para no doble-contar el dia de valoracion
                aux_tools.calcular_dias_diferencia(
                    pl.col("fecha_fin_devengo"),
                    pl.col("fecha_valoracion"),
                    incluir_extremos=False,
                ).map_elements(lambda x: max(x, 0))
            )
            .when(
                (pl.col("estado_devengo") == "finalizado")
                | (pl.col("estado_devengo") == "entra_devengado")
            )
            .then(pl.lit(0))
            .alias("dias_no_devengados")
        )
        .with_columns(
            # validacion del calculo de dias
            (
                pl.col("dias_devengados") + pl.col("dias_no_devengados")
                == pl.col("dias_constitucion")
            ).alias("control_suma_dias")
        )
        .with_columns(
            # valor diario devengo (prima diaria en sap)
            (pl.col("valor_base_devengo") / pl.col("dias_constitucion")).alias(
                "valor_devengo_diario"
            )
        )
        .with_columns(
            (  # el valor que falta por devengarse es el saldo
                pl.col("valor_devengo_diario") * pl.col("dias_no_devengados")
            ).alias("saldo")
        )
        .with_columns(
            pl.when(
                (pl.col("fecha_constitucion") <= pl.col("fecha_valoracion"))
                & (pl.col("fecha_inicio_periodo") <= pl.col("fecha_constitucion"))
            )
            .then(pl.col("dias_constitucion") * pl.col("valor_devengo_diario"))
            .otherwise(pl.lit(0.0))
            .alias("valor_constitucion")
        )
        .with_columns(
            # Queremos que si entra devengado, libere todo
            pl.when(
                (pl.col("estado_devengo") == "entra_devengado")
                & (pl.col("valor_constitucion") != 0)
            )
            .then(pl.col("dias_constitucion"))
            .when(pl.col("estado_devengo") == "no_iniciado")
            .then(pl.lit(0))
            .when(pl.col("fecha_inicio_periodo") <= pl.col("fecha_fin_devengo"))
            .then(
                aux_tools.calcular_dias_diferencia(
                    pl.min_horizontal(
                        [pl.col("fecha_valoracion"), pl.col("fecha_fin_devengo")]
                    ),
                    pl.when(
                        pl.col("fecha_constitucion") > pl.col("fecha_inicio_periodo")
                    )
                    .then(pl.col("fecha_inicio_vigencia"))
                    .otherwise(pl.col("fecha_inicio_periodo")),
                )
            )
            .otherwise(pl.lit(0))
            .alias("dias_liberacion")
        )
        .with_columns(
            (pl.col("dias_liberacion") * pl.col("valor_devengo_diario")).alias(
                "valor_liberacion"
            )
        )
        .with_columns(
            (pl.col("dias_devengados") * pl.col("valor_devengo_diario")).alias(
                "valor_liberacion_acum"
            )
        )
    )

    return output_deveng_diario


def deveng_cincuenta(
    input_deveng_cinq: pl.DataFrame, fe_valoracion: dt.date
) -> pl.DataFrame:
    """
    Recibe un input preprocesado para devengo y devuelve el devengamiento segun las reglas del 50/50
    """

    mes_fin_vigencia = aux_tools.yyyymm(pl.col('fecha_fin_devengo'))
    mes_constitucion = aux_tools.yyyymm(pl.col("fecha_constitucion"))
    
    entra_devengado = mes_constitucion > mes_fin_vigencia

    es_periodo_constit = (
        pl.col("fecha_constitucion") <= pl.col("fecha_valoracion")
    ) & (pl.col("fecha_inicio_periodo") <= pl.col("fecha_constitucion"))
    # aplica las condiciones para constituir
    output_deveng_cinq = input_deveng_cinq.with_columns(
        pl.when(es_periodo_constit)
        .then(pl.col("valor_base_devengo"))
        .otherwise(0.0)
        .alias("valor_constitucion")
    ).with_columns(mes_constitucion.alias("mes_constitucion"))

    # libera solo a cierre de mes -> si no es cierre la norma me obliga a mantener el 50%
    es_cierre_mes = pl.lit(aux_tools.es_ultimo_dia_mes(fe_valoracion))
    # primera liberacion ocurre el max mes entre mes constitucion y mes inicio devengo
    aux_fe_primera_lib = pl.col("fecha_inicio_devengo")
    mes_primera_lib = aux_tools.yyyymm(aux_fe_primera_lib)
    mes_segunda_lib = aux_tools.yyyymm(aux_fe_primera_lib.dt.offset_by("1mo"))

    mes_valoracion = aux_tools.yyyymm(pl.col("fecha_valoracion"))
    # calcula la liberacion por meses
    lib_mes_actual = (
        pl.when(entra_devengado)
        .then(pl.col("valor_base_devengo") * es_periodo_constit)
        .when(
            ((mes_valoracion == mes_primera_lib) | (mes_valoracion == mes_segunda_lib))
            & es_cierre_mes
        )
        .then(pl.col("valor_base_devengo") * 0.5)
        .otherwise(0.0)
    )
    # para conocer el saldo debo acumular la liberacion que solo se puede dar en max 2 periodos
    lib_acumulada = (
        pl.when(
            (mes_valoracion > mes_segunda_lib) | entra_devengado
        )
        .then(pl.col("valor_base_devengo"))
        .when(
            ((mes_valoracion == mes_segunda_lib) & es_cierre_mes)
        )
        .then(pl.col("valor_base_devengo"))
        .when(
            ((mes_valoracion == mes_segunda_lib) & ~es_cierre_mes)
        )
        .then(pl.col("valor_base_devengo") * 0.5)
        .when(
            (mes_valoracion == mes_primera_lib)
            & es_cierre_mes
        )
        .then(pl.col("valor_base_devengo") * 0.5)
        .otherwise(0.0)
    )
    # el estado segun las fechas
    devengo_no_iniciado = pl.col("fecha_valoracion") < pl.col("fecha_inicio_devengo")
    devengo_en_curso = (pl.col("fecha_inicio_devengo") <= pl.col('fecha_valoracion')) & (
        pl.col('fecha_valoracion') < pl.col("fecha_inicio_devengo").dt.offset_by('1mo').dt.month_end()
    )

    output_deveng_cinq = (
        output_deveng_cinq.with_columns(
            pl.when(entra_devengado)
            .then(pl.lit("entra_devengado"))
            .when(devengo_no_iniciado)
            .then(pl.lit("no_iniciado"))
            .when(devengo_en_curso)
            .then(pl.lit("en_curso"))
            .otherwise(pl.lit("finalizado"))
            .alias("estado_devengo")
        )
        .with_columns(
            # El saldo es el 100% si no ha empezado
            pl.when(devengo_no_iniciado)
            .then(pl.col("valor_base_devengo"))
            # el 100% menos la lib acumulada si está liberando
            .when(devengo_en_curso)
            .then(pl.col("valor_base_devengo") - lib_acumulada)
            # sino, ya acabó y el saldo es cero
            .otherwise(0)
            .alias("saldo")
        )
        .with_columns(lib_mes_actual.alias("valor_liberacion"))
        .with_columns(mes_primera_lib.alias("mes_ini_liberacion"))
        .with_columns(mes_segunda_lib.alias("mes_fin_liberacion"))
        .with_columns(lib_acumulada.alias("valor_liberacion_acum"))
    )

    return output_deveng_cinq


def devengo_diario_vs_limite(input_costo: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica el devengo del costo de contrato de RA no prop tomando el máximo entre devengo diario
    y consumo del límite agregado del contrato
    """
    # Expresiones de saldo segun si es recibo nuevo o ya viene devengandose
    saldo_anterior = (
        pl.col("saldo_anterior")
        if "saldo_anterior" in input_costo.columns
        else pl.col("valor_base_devengo")
    )

    # Devengo diario es la base
    output_devengo_costo = (
        deveng_diario(input_costo)
        .with_columns(
            # % consumo del límite en el mes
            (
                (
                    pl.col("valor_siniestros_incurridos_mes")
                    + pl.col("valor_salvamentos_mes")
                )
                / pl.col("limite_agregado_valor_instalado")
            ).alias("porc_consumo_limite")
        )
        .with_columns(
            # valor de liberación por límite
            (pl.col("valor_base_devengo") * pl.col("porc_consumo_limite")).alias(
                "valor_liberacion_limite"
            )
        )
        .with_columns(
            # elegir el método de liberación: diario o límite
            pl.when(
                aux_tools.yyyymm(pl.col("fecha_valoracion"))
                == aux_tools.yyyymm(pl.col("fecha_fin_devengo"))
            )  # último mes debe liberar todo el saldo
            .then(pl.lit("saldo_restante"))
            .when(pl.col("valor_liberacion_limite") > pl.col("valor_liberacion"))
            .then(pl.lit("consumo_limite"))
            .otherwise(pl.lit("diario"))
            .alias("regla_devengo")
        )
        .with_columns(
            # aplicar el método de liberación elegido
            pl.when(pl.col("regla_devengo") == "saldo_restante")
            .then(saldo_anterior)
            .when(pl.col("regla_devengo") == "consumo_limite")
            .then(pl.col("valor_liberacion_limite"))
            .otherwise(pl.col("valor_liberacion"))
            .alias("valor_liberacion")
        )
        .with_columns(
            # acumulado de liberación recalculado
            (
                (pl.col("valor_base_devengo") - saldo_anterior)
                + pl.col("valor_liberacion")
            ).alias("valor_liberacion_acum")
        )
        .with_columns(
            # nuevo saldo después de la liberación (esto es antes de que se apliquen signos de reserva)
            (saldo_anterior - pl.col("valor_liberacion")).alias("saldo")
        )
    )

    return output_devengo_costo

def devengo_componente_inversion(
        input_deveng_comp_inv: pl.DataFrame
) -> pl.DataFrame:
    """
    Recibe un input preprocesado de devengamiento y devuelve el devengo del componente de inversión, 
    el cual técnicamente no es un devengo pues va acumulando dinero para liberar cuando se cancele 
    la póliza o se llegue a la fecha de fin de vigencia de ésta
    """
    output_deveng_comp_inv = (
        input_deveng_comp_inv
        .with_columns([
            aux_tools.yyyymm(pl.col("fecha_valoracion")).alias("mes_valoracion"),
            aux_tools.yyyymm(pl.col("fecha_constitucion")).alias("mes_constitucion"),
            aux_tools.yyyymm(pl.col("fecha_fin_devengo")).alias("mes_fin_devengo"),
        ])
        .with_columns([
            pl.when(
                (pl.col('mes_valoracion') == pl.col('mes_constitucion')) &
                (pl.col('fecha_valoracion') >= pl.col('fecha_constitucion'))
            )
            .then(pl.col('valor_base_devengo'))
            .otherwise(pl.lit(0.0))
            .alias('valor_constitucion'),
            
            pl.when(
                (pl.col("fecha_valoracion") >= pl.col("fecha_fin_devengo"))
            )
            .then(pl.col("valor_base_devengo"))
            .otherwise(pl.lit(0.0))
            .alias("valor_liberacion_acum"),
            
            pl.when(
                (pl.col('mes_valoracion') == pl.col('mes_fin_devengo')) &
                (pl.col('fecha_valoracion') >= pl.col('fecha_fin_devengo'))
            )
            .then(pl.col('valor_base_devengo'))
            .otherwise(pl.lit(0.0))
            .alias("valor_liberacion"),
        ])
        .with_columns([
            (pl.col("valor_base_devengo") - pl.col("valor_liberacion_acum"))
            .alias("saldo")
        ])
    )
    
    return output_deveng_comp_inv

def etiquetar_resultado_devengo(data_devengo: pl.DataFrame) -> pl.DataFrame:
    """
    calcula campos necesarios según los inputs y outputs del devengo
    y organiza el output con estas columnas incluidas
    """
    # condiciones para definir los niveles de agregacion
    es_anio_actual = (
        pl.col("fecha_constitucion").dt.year() == pl.col("fecha_valoracion").dt.year()
    )
    # aplica condiciones
    data_devengo_out = (
        data_devengo.pipe(aux_tools.agregar_cohorte_dinamico)
        .with_columns(
            # define si la liberacion es del año actual o anterior
            pl.when(es_anio_actual)
            .then(pl.lit("anio_actual"))
            .otherwise(pl.lit("anio_anterior"))
            .alias("anio_liberacion")
        )
        .pipe(aux_tools.etiquetar_transicion)
    )

    return data_devengo_out


import polars as pl

def devengo_comp_financiacion(input_prep_cf: pl.DataFrame) -> pl.DataFrame:
    """
    Realiza calculos actuariales de reserva para el caso de PAA con componente de financiacion.
    El input debe contener las columnas financieras de ipc y factores de curva de interes lir necesarios.
    """
    # el estado segun las fechas
    devengo_no_iniciado = pl.col("fecha_valoracion") < pl.col("fecha_inicio_devengo")
    # Se incluye el mes final para permitir el último movimiento de liberación e intereses
    devengo_en_curso = (pl.col("fecha_inicio_devengo") <= pl.col('fecha_valoracion')) & (pl.col('mes_valoracion') <= pl.col('mes_fin_vigencia'))
    es_periodo_ini = pl.col('mes_valoracion') == pl.col('mes_inicio_vigencia')

    out_devengo_financiacion = (
        input_prep_cf
        .with_columns(
            # Estado del devengo a la fecha de valoracion por trazabilidad
            pl.when(devengo_no_iniciado)
            .then(pl.lit("no_iniciado"))
            .when(devengo_en_curso)
            .then(pl.lit("en_curso"))
            .otherwise(pl.lit("finalizado"))
            .alias("estado_devengo")
        )
        .with_columns(
            # Ponderadores por días exactos de los nodos extremos (todos los del centro son 1.00)
            (pl.col('dias_vig_ini') / pl.col('dias_nodo_ini')).alias('peso_nodo_ini'),
            (pl.col('dias_vig_fin') / pl.col('dias_nodo_fin')).alias('peso_nodo_fin'),
        )
        .with_columns(
            # El peso de los dias del nodo actual y nodo anterior
            pl.when(pl.col('mes_valoracion') == pl.col('mes_inicio_vigencia'))
            .then(pl.col('peso_nodo_ini'))
            .when(pl.col('mes_valoracion') == pl.col('mes_fin_vigencia'))
            .then(pl.col('peso_nodo_fin'))
            .otherwise(pl.lit(1.0))
            .alias('peso_nodo_actual'),
            
            pl.when(pl.col('mes_valoracion_anterior') == pl.col('mes_inicio_vigencia'))
            .then(pl.col('peso_nodo_ini'))
            .when(pl.col('mes_valoracion_anterior') == pl.col('mes_fin_vigencia'))
            .then(pl.col('peso_nodo_fin'))
            .otherwise(pl.lit(1.0))
            .alias('peso_nodo_anterior'),
        )
        .with_columns(
            # suma de factores de descuento total curva reponderando los dos extremos
            (
                pl.col('sum_desc_lir_nodo_fin') 
                - pl.col('desc_lir_nodo_ini') * (1 - pl.col('peso_nodo_ini')) # Ajuste inicio
                - pl.col('desc_lir_nodo_fin') * (1 - pl.col('peso_nodo_fin')) # Ajuste fin
            ).alias('suma_factores_total')
        )
        .with_columns(
            # solo si estamos en mes ini o fin se repondera el factor de acumuulacion de intereses
            pl.when((es_periodo_ini) | (pl.col('mes_valoracion') == pl.col('mes_fin_vigencia')))
            .then(pl.col('fact_acum_val') * (1 + pl.col('tasa_fwd_real_val')).pow(pl.col('peso_nodo_actual') - 1))
            .otherwise(pl.col('fact_acum_val'))
            .alias('factor_cap_real'),
            
            # y el anterior...
            pl.when((pl.col('mes_valoracion_anterior') == pl.col('mes_inicio_vigencia')) | (pl.col('mes_valoracion_anterior') == pl.col('mes_fin_vigencia')))
            .then(pl.col('fact_acum_ant') * (1 + pl.col('tasa_fwd_real_ant')).pow(pl.col('peso_nodo_anterior') - 1))
            .otherwise(pl.col('fact_acum_ant'))
            .alias('factor_cap_real_ant')
        )
        .with_columns (
            # suma de factores a la fecha de valoracion reponderando extremos necesarios
            pl.when(pl.col('mes_valoracion') >= pl.col('mes_fin_vigencia'))
            .then(pl.col('suma_factores_total'))    # si se valora en el ultimo nodo es igual al total
            .otherwise( # Si está en curso solo se repondera el nodo inicial
                pl.col('sum_desc_lir_val') - pl.col('desc_lir_nodo_ini') * (1 - pl.col('peso_nodo_ini')))
            .alias('sum_desc_valoracion'),
            
            # lo mismo para el periodo anterior
            pl.when(pl.col('mes_valoracion_anterior') >= pl.col('mes_fin_vigencia'))
            .then(pl.col('suma_factores_total'))
            .when(pl.col('mes_valoracion_anterior') < pl.col('mes_inicio_vigencia'))
            .then(pl.lit(0.0))
            .otherwise(pl.col('sum_desc_lir_ant') - pl.col('desc_lir_nodo_ini') * (1 - pl.col('peso_nodo_ini')))
            .alias('sum_desc_anterior')
        )
        .with_columns(
            # sumatoria Futura de lo que queda por devengar desde t+1
            (pl.col('suma_factores_total') - pl.col('sum_desc_anterior')).alias('suma_factores_remanente_ant'),
            (pl.col('suma_factores_total') - pl.col('sum_desc_valoracion')).alias('suma_factores_remanente')
        )
        .with_columns(          
            # indice relativo de ipc fecha actual y anterior reponderando el primer mes por el peso de los dias
            ( ( pl.col('indice_ipc_actual')/pl.col('indice_ipc_ini') ) * ( ( 1 + pl.col('tasa_ipc_ini') ).pow( pl.col('peso_nodo_ini') ) ) ).alias('factor_ajuste_ipc'),
            ( (pl.col('indice_ipc_anterior')/pl.col('indice_ipc_ini') ) * ( ( 1 + pl.col('tasa_ipc_ini') ).pow( pl.col('peso_nodo_ini') ) ) ).alias('factor_ajuste_ipc_ant'),
            # tasa nominal para acreditacion de intereses
            (( (1 + pl.col('tasa_fwd_real_val')) * (1 + pl.col('tasa_ipc_actual')) ).pow(pl.col('peso_nodo_actual')) - 1).alias('tasa_acreditacion'),
        )
        .with_columns(
            # calculo del saldo de reserva anterior y del saldo actual (en valor absoluto)
            pl.when(devengo_en_curso)
            .then( (pl.col('valor_base_devengo') * pl.col('factor_ajuste_ipc_ant') * pl.col('factor_cap_real_ant') * pl.col('suma_factores_remanente_ant') / pl.col('suma_factores_total')).abs() )
            .otherwise(pl.lit(0.0))
            .alias('saldo_anterior'),
            
            # saldo actual
            pl.when(devengo_en_curso)
            .then( (pl.col('valor_base_devengo') * pl.col('factor_ajuste_ipc') * pl.col('factor_cap_real') * pl.col('suma_factores_remanente') / pl.col('suma_factores_total')).abs() )
            .otherwise(pl.lit(0.0))
            .alias('saldo')
        )
        .with_columns(
            pl.when(es_periodo_ini).then(pl.col('valor_base_devengo').abs()).otherwise(pl.lit(0.0)).alias('valor_constitucion')
        )
        .with_columns(
            # calculo de liberacion y acreditacion de intereses a tasa nominal (en valor absoluto)
            pl.when(devengo_en_curso)
            .then( (pl.col('saldo_anterior') * pl.col('tasa_acreditacion')).abs() )
            .otherwise(pl.lit(0.0))
            .alias('acreditacion_intereses')
        )
        .with_columns(
            # la liberacion del periodo
            pl.when(devengo_en_curso)
            .then((pl.col('saldo_anterior') + pl.col('acreditacion_intereses') - pl.col('saldo') + pl.col('valor_constitucion')).abs())
            .otherwise(pl.lit(0.0))
            .alias('valor_liberacion'),
            # la liberacion acumulada por trazabilidad (es compleja de calcular y no requerida)
            pl.lit(None).cast(pl.Float64).alias('valor_liberacion_acum')
        )
    )
    return out_devengo_financiacion



def devengar(input_deveng: pl.DataFrame, fe_valoracion: dt.date) -> pl.DataFrame:
    """
    Recibe cualquier input preprocesado para devengamiento
    devuelve el output de devengo consolidado y organizado relativo a la fecha de valoración
    integra las distintas reglas de devengo, las aplica según corresponda y consolida el resultado
    """

    # parametros generales segun la fecha de valoracion
    input_devengo = (
        input_deveng.with_columns(pl.lit(fe_valoracion).alias("fecha_valoracion"))
        .with_columns(
            # por defecto los mov se calculan para el periodo desde el inicio del mes de la fe valoracion
            pl.col("fecha_valoracion").dt.month_start().alias("fecha_inicio_periodo")
        )
        .with_columns(
            # define fecha cierre anterior para el delta
            pl.col("fecha_valoracion")
            .dt.offset_by("-1mo")
            .dt.month_end()
            .alias("fecha_valoracion_anterior")
        )
    )
    # define si aplica componente de financiacion
    aplica_financiacion = pl.col('aplica_comp_financ') == 1
    # define si es componente de inversión
    aplica_comp_inv = pl.col('tipo_insumo') == 'componente_inversion_directo'
    # define si aplica devengo de costo contrato
    aplica_costo_contrato = pl.col("tipo_insumo").is_in(
        ["costo_contrato_rea_noprop", "recup_onerosidad_np"]
    )
    # define si aplica 5050 y hace la particion del insumo
    dias_devengo = aux_tools.calcular_dias_diferencia(
        pl.col("fecha_fin_devengo"), pl.col("fecha_inicio_devengo")
    )
    aplica_5050 = (dias_devengo <= 32) & (pl.col("candidato_devengo_50_50") == 1)
    input_devengo = input_devengo.with_columns(
        pl.when(aplica_comp_inv)
        .then(pl.lit("componente_inversion"))
        .when(aplica_5050)
        .then(pl.lit("regla_50_50"))
        .when(aplica_financiacion)
        .then(pl.lit('componente_financiacion'))
        .otherwise(pl.lit("diario"))
        .alias("regla_devengo")
    )
    
    input_devengo_comp_financiacion = input_devengo.filter(aplica_financiacion)
    input_devengo_comp_inv = input_devengo.filter(aplica_comp_inv)
    input_devengo_5050 = input_devengo.filter(aplica_5050 & (~aplica_costo_contrato) & (~aplica_comp_inv))
    input_devengo_costcon = input_devengo.filter(aplica_costo_contrato & (~aplica_5050) & (~aplica_comp_inv))
    input_devengo_diario = input_devengo.filter(
        (~aplica_5050) & (~aplica_costo_contrato) & (~aplica_comp_inv) & (~aplica_financiacion)
    )

    # se inicializan outputs vacío y campos output como copia de la lista (porque si no modifica la original)
    outputs, campos_output = [], params.CAMPOS_OUTPUT_CONTABLE.copy()
    # aplica el devengamiento a cada particion solo si tiene datos de entrada
    if input_devengo_comp_financiacion.height > 0:
        outputs.append(devengo_comp_financiacion(input_devengo_comp_financiacion))
        campos_output.extend(params.CAMPOS_OUTPUT_FINANCIACION)
    if input_devengo_comp_inv.height > 0:
        outputs.append(devengo_componente_inversion(input_devengo_comp_inv))
    if input_devengo_diario.height > 0:
        outputs.append(deveng_diario(input_devengo_diario))
        campos_output.append(params.CAMPOS_OUTPUT_DIARIO)
    if input_devengo_5050.height > 0:
        outputs.append(
            deveng_cincuenta(input_devengo_5050, fe_valoracion=fe_valoracion)
        )
        campos_output.extend(params.CAMPOS_OUTPUT_5050)
    if input_devengo_costcon.height > 0:
        outputs.append(devengo_diario_vs_limite(input_devengo_costcon))
        campos_output.extend(params.CAMPOS_OUTPUT_LIMITE)
    if not outputs:
        return pl.DataFrame()

    # retorna un consolidado tipo union all de los outputs
    output_devengo_consolidado = (
        pl.concat(outputs, how="diagonal")
        .with_columns(
            [
                # aplica el signo de reserva segun tipo insumo y movimientos
                (pl.col("valor_constitucion") * pl.col("signo_constitucion")).alias(
                    "valor_constitucion"
                ),
                (pl.col("valor_liberacion") * -1 * pl.col("signo_constitucion")).alias(
                    "valor_liberacion"
                ),
                (
                    pl.col("valor_liberacion_acum") * -1 * pl.col("signo_constitucion")
                ).alias("valor_liberacion_acum"),
                (pl.col("saldo") * pl.col("signo_constitucion")).alias("saldo"),
                # El signo de la acreditación de intereses siempre es el mismo de la constitución
                pl.when(pl.col("acreditacion_intereses").is_not_nan())
                .then(pl.col("acreditacion_intereses") * pl.col("signo_constitucion"))
                .otherwise(pl.lit(None))
                .alias("acreditacion_intereses"),
            ]
        )
        .with_columns(
            (pl.col("saldo") 
             - pl.col("valor_constitucion")
             - pl.col("valor_liberacion")
            ).alias("saldo_anterior"),
        )
        .pipe(etiquetar_resultado_devengo)
    )

    campos_output.extend(params.CAMPOS_OUTPUT_CALCULADO)
    campos_output = list(dict.fromkeys(campos_output))
    campos_input = [
        col for col in output_devengo_consolidado.columns if col not in campos_output
    ]

    return output_devengo_consolidado.select(campos_input + campos_output)
