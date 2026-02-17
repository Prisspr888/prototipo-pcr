from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path

base_dir = Path(__file__).resolve().parent


# saltar una fila al leer los insumos que se usa como paralelo de los nombres de campos en las HU
SKIP_ROWS = 1

# Ubicación de los insumos relativa al script de ejecución
RUTA_INSUMOS = base_dir.parent / "inputs" / "insumos.xlsx"
RUTA_REL_BT = base_dir.parent / "inputs" / "relacion_bt.xlsx"
RUTA_NOMENCLATURA = base_dir.parent / "inputs" / "nomenclatura.xlsx"
RUTA_PRODUCCION_ARL = base_dir.parent / "inputs" / "produccion_arl.xlsx"
RUTA_COSTO_CONTRATO_ARL = base_dir.parent / "inputs" / "costo_contrato_arl.xlsx"
RUTA_CARTERA_ARL = base_dir.parent / "inputs" / "cartera_arl.xlsx"
RUTA_CUENTA_CORRIENTE = base_dir.parent / "inputs" / "cuenta_corriente.xlsx"
RUTA_CUENTA_CORRIENTE_ARL = base_dir.parent / "inputs" / "cuenta_corriente_arl.xlsx"
RUTA_GASTOS = base_dir.parent / "inputs" / "gastos - tests.xlsx"
RUTA_CAMARA_SOAT = base_dir.parent / "inputs" / "camara_soat.xlsx"

# Insumos transversales
HOJA_PARAMETROS_CONTAB = "param_contabilidad"
HOJA_EXCEPCIONES_50_50 = "excepciones_50_50"
HOJA_MONEDA = "tasa_cambio"
HOJA_DESCUENTO = "porc_descuento"
HOJA_TIPO_SEGURO = "tipo_seguro"
HOJA_CARTERA = "cuenta_cobrar_directo"
HOJA_CUENTAS_REA = "cuentas_reaseguro"
RUTA_RIESGO_CREDITO = base_dir.parent / "inputs" / "riesgo_credito.xlsx"

# Insumos de recibos contabilizados
HOJA_PDN = "produccion_directo"
HOJA_COMP_INV = "componente_inversion_directo"
HOJA_CESION = "produccion_rea"
HOJA_COMISION_REA = "comision_rea"
HOJA_COSTO_CONTRATO = "costo_rea_noprop"
HOJA_SEGUIMIENTO_REA = "seguimiento_rea_noprop"

# Insumos de onerosidad
HOJA_ONEROSIDAD = "onerosidad"
HOJA_RECUP_ONEROSIDAD = "recup_onerosidad"


# Fechas relevantes para cada ejecución
FECHA_VALORACION = date(2025, 9, 30)
FECHA_TRANSICION = date(2024, 12, 31)
FECHA_VALORACION_ANTERIOR = FECHA_VALORACION + relativedelta(months=-1)

# Rutas salidas
RUTA_SALIDA_DEVENGO = (
    base_dir.parent
    / "output"
    / f"output_devengo_fluc_{FECHA_VALORACION.strftime('%d%m%Y')}.xlsx"
)
RUTA_SALIDA_CONTABLE = (
    base_dir.parent
    / "output"
    / f"output_contable_{FECHA_VALORACION.strftime('%d%m%Y')}.xlsx"
)

# Parametros generales
NIVELES_DETALLE = ["recibo", "cobertura"]
MONEDA_DESTINO = "COP"

# Ayudan a controlar campos de salida, orden y nombres
CAMPOS_OUTPUT_CONTABLE = [
    # Campos necesarios para contabilizar, deben aparecer en el output con estos nombres
    "componente",
    "tipo_contabilidad",
    "regla_devengo",  # nivel de agregacion y parametros
    "tipo_contrato",
    "cohorte",
    "anio_liberacion",
    "transicion",  # nivel de agregacion
    "fecha_inicio_periodo",
    "fecha_valoracion",  # fecha de inicio y fin del periodo de valoracion
    "fecha_constitucion",
    "fecha_inicio_devengo",
    "fecha_fin_devengo",
    "valor_base_devengo",  # parametros devengo
]
CAMPOS_OUTPUT_DIARIO = [
    # solo aplican para devengo diario - se separan
    "dias_devengados",
    "dias_no_devengados",
    "control_suma_dias",  
    "dias_constitucion",
    "dias_liberacion",
    "valor_devengo_diario",
]
CAMPOS_OUTPUT_5050 = [
    "mes_constitucion",
    "mes_ini_liberacion",
    "mes_fin_liberacion",  # solo estan para devengo 5050
]
CAMPOS_OUTPUT_LIMITE = [
    "porc_consumo_limite",  # solo estan para devengo por consumo del limite
]
CAMPOS_OUTPUT_CALCULADO = [
    "estado_devengo",
    "valor_constitucion",
    "valor_liberacion",
    "valor_liberacion_acum",
    "saldo",  # movimientos y saldo
]
CAMPOS_OUTPUT_FINANCIACION = [
    # Trazabilidad de insumos
    "pais_curva",
    "moneda_curva",
    # Variables formula saldo: base_devengo * factor_ajuste_ipc * factor_cap_real * (remanente / total)
    "factor_ajuste_ipc",       
    "factor_cap_real",         
    "suma_factores_remanente", 
    "suma_factores_total",     
    # Variables del Movimiento de Intereses
    "saldo_anterior",          
    "tasa_acreditacion",       
    "acreditacion_intereses",

    # campos para test
    #"peso_nodo_ini",
    #"peso_nodo_fin",
    #"tasa_ipc_actual",
    #"tasa_fwd_real_val",
    #"saldo_anterior",
    #'factor_ajuste_ipc_ant',
    #'factor_cap_real_ant',
]


# a qué tipos de contabilidad aplica deterioro
APLICA_DETERIORO = ["ifrs17_local", "ifrs_17_corporativo"]


# Las columnas que se deben pivotear para el output contable
COLUMNAS_CALCULO = [
    "valor_constitucion",
    "valor_liberacion",
    "saldo",
    "fluctuacion_liberacion",
    "fluctuacion_constitucion",
    "constitucion_deterioro",
    "liberacion_deterioro",
]

# Ramos que se postean como reserva matematica
RAMOS_MATEMATICA = ["092", "095", "196"]
