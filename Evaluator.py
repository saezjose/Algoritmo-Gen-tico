"""
Módulo 3: Evaluación y Función Objetivo J(x)
==============================================

Calcula todos los componentes de la función objetivo total:

    J(x) = D(x) + tau(x) + P_Q(x) + P_C(x) + P_R(x) + P_A(x)
           + P_prem(x) + P_inv(x)

y el fitness asociado phi(x) = -J(x).

Notas de diseño e interpretación (documentadas para la defensa oral)
----------------------------------------------------------------------
1) Penalización global de invalidez P_inv(x): la pauta escrita en prosa
   ("Pauta de trabajo y rúbrica", ecuación completa) define
   P_inv(x) = 10.000 * I_inv(x). Se adopta ese valor (10.000) como
   constante por defecto, ya que corresponde a la fuente matemática
   completa y auditable. El valor es un parámetro nombrado
   (PENALIZACION_GLOBAL_INVALIDA) fácilmente ajustable si la cátedra confirma
   un valor distinto (p. ej. 100.000).

2) Bloques de giros (P_R): un bloque se cierra (se agrega a B(x)) cada
   vez que ocurre un M exitoso (el individuo cambia de celda). Además,
   se cierra también el bloque remanente al finalizar la ejecución del
   cromosoma, aunque no haya un M exitoso posterior. Esto evita que una
   secuencia de giros ubicada al final del cromosoma quede sin penalizar
   por el simple hecho de no ir seguida de un movimiento.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from cargador_laberinto import Laberinto, Posicion
from simulador import Cromosoma, ResultadoSimulacion, RegistroPaso

# --- Constantes de penalización (parametrizadas como constantes con nombre) ---
PENALIZACION_PAUSA_INTERMEDIA = 10        # P_Q: por cada Q intermedio
PENALIZACION_CHOQUE = 30                  # P_C: por cada choque (M fallido)
PENALIZACION_ACCION_POST_META = 100       # P_A: por cada acción activa tras la meta
PENALIZACION_DETENCION_PREMATURA_POR_Q = 10  # P_prem: por cada Q de la cola final (si inválido)
PENALIZACION_GLOBAL_INVALIDA = 10_000     # P_inv: penalización global si no es solución válida


def penalizacion_bloque_giro(b: int) -> int:
    """f(b): penalización de un bloque de giros de longitud b.

    f(b) = 0        si b <= 1
         = 10       si b == 2
         = 30       si b == 3
         = 120(b-3) si b >= 4
    """
    if b <= 1:
        return 0
    if b == 2:
        return 10
    if b == 3:
        return 30
    return 120 * (b - 3)


@dataclass(frozen=True)
class ResultadoEvaluacion:
    """Resultado completo de evaluar un cromosoma sobre un laberinto."""

    cromosoma: Cromosoma
    n: int
    posicion_final: Posicion
    llegada: Posicion

    D: int                        # Distancia de Manhattan final
    llegadas: Tuple[int, ...]     # T_z(x): pasos (1-indexados) de llegada efectiva
    l: int | None                 # ℓ(x): última llegada efectiva (o None)
    tau: int                      # τ(x)
    es_valido: bool               # Solución válida completa
    rho: int                      # 0, 1 o 2 (prioridad de factibilidad)

    conteo_Q_intermedio: int
    conteo_choques: int
    bloques_giro: Tuple[int, ...]
    conteo_acciones_post_meta: int
    conteo_Q_prematuro: int

    P_Q: float
    P_C: float
    P_R: float
    P_A: float
    P_prem: float
    P_inv: float

    J: float
    phi: float

    trayectoria: Tuple[RegistroPaso, ...]

    def clave_ranking(self) -> Tuple[int, float, int, int]:
        """Clave lexicográfica (rho(x), J(x), D(x), tau(x)) para el ranking evolutivo."""
        return (self.rho, self.J, self.D, self.tau)


def _calcular_distancia_manhattan(posicion_final: Posicion, llegada: Posicion) -> int:
    return abs(posicion_final[0] - llegada[0]) + abs(posicion_final[1] - llegada[1])


def _calcular_llegadas(trayectoria: Tuple[RegistroPaso, ...], llegada: Posicion) -> Tuple[int, ...]:
    """T_z(x) = {t : p_{t-1} != z ^ p_t = z}."""
    return tuple(
        registro.indice_paso
        for registro in trayectoria
        if registro.posicion_antes != llegada and registro.posicion_despues == llegada
    )


def _calcular_tau(
    llegadas: Tuple[int, ...], trayectoria: Tuple[RegistroPaso, ...], n: int
) -> Tuple[int, bool, int | None]:
    """Calcula (tau(x), es_valido, ell(x))."""
    if not llegadas:
        return n + 1, False, None

    valor_l = max(llegadas)
    if valor_l < n and all(trayectoria[k].gen == "Q" for k in range(valor_l, n)):
        return valor_l, True, valor_l
    return n + 1, False, valor_l


def _calcular_pausas_intermedias(cromosoma: Cromosoma) -> int:
    """Q_int(x) = #{k in {1,...,n-1} : g_k = Q ^ exists j>k con g_j != Q}."""
    n = len(cromosoma)
    ultimo_indice_no_q: int | None = None
    for idx in range(n - 1, -1, -1):
        if cromosoma[idx] != "Q":
            ultimo_indice_no_q = idx
            break
    if ultimo_indice_no_q is None:
        return 0  # todos los genes son Q: no existe "parte activa" posterior
    return sum(1 for idx in range(0, ultimo_indice_no_q) if cromosoma[idx] == "Q")


def _calcular_choques(trayectoria: Tuple[RegistroPaso, ...]) -> int:
    return sum(1 for registro in trayectoria if registro.choco)


def _calcular_bloques_giro(trayectoria: Tuple[RegistroPaso, ...]) -> Tuple[int, ...]:
    """Construye el multiconjunto B(x) de longitudes de bloques de giros.

    Un bloque se compone de acciones H/A ejecutadas en la misma celda.
    Q y M fallido no modifican el contador del bloque. Un M exitoso cierra
    el bloque (lo agrega a B(x)) y reinicia el contador a 0. El bloque
    remanente al finalizar el cromosoma también se cierra (ver docstring
    del módulo).
    """
    bloques: List[int] = []
    bloque_actual = 0
    for registro in trayectoria:
        if registro.gen in ("H", "A"):
            bloque_actual += 1
        elif registro.gen == "M" and not registro.choco:
            bloques.append(bloque_actual)
            bloque_actual = 0
        # Q y M fallido: no modifican bloque_actual
    if bloque_actual > 0:
        bloques.append(bloque_actual)
    return tuple(bloques)


def _calcular_acciones_post_meta(trayectoria: Tuple[RegistroPaso, ...], llegada: Posicion) -> int:
    """A_meta(x): acciones activas (H, A, M) ejecutadas ya estando en la meta."""
    return sum(
        1
        for registro in trayectoria
        if registro.posicion_antes == llegada and registro.gen in ("H", "A", "M")
    )


def _calcular_detencion_prematura(cromosoma: Cromosoma, es_valido: bool) -> int:
    """Q_prem(x): longitud de la cola final de genes Q, solo si la solución es inválida."""
    if es_valido:
        return 0
    conteo = 0
    for gen in reversed(cromosoma):
        if gen == "Q":
            conteo += 1
        else:
            break
    return conteo


def evaluar_cromosoma(
    cromosoma: Cromosoma, resultado_sim: ResultadoSimulacion, laberinto: Laberinto
) -> ResultadoEvaluacion:
    """Evalúa completamente un cromosoma ya simulado, calculando J(x) y phi(x)."""
    n = len(cromosoma)
    llegada = laberinto.llegada
    trayectoria = resultado_sim.trayectoria

    D = _calcular_distancia_manhattan(resultado_sim.posicion_final, llegada)
    llegadas = _calcular_llegadas(trayectoria, llegada)
    tau, es_valido, valor_l = _calcular_tau(llegadas, trayectoria, n)

    if es_valido:
        rho = 0
    elif llegadas:
        rho = 1
    else:
        rho = 2

    conteo_q_intermedio = _calcular_pausas_intermedias(cromosoma)
    P_Q = PENALIZACION_PAUSA_INTERMEDIA * conteo_q_intermedio

    conteo_choques = _calcular_choques(trayectoria)
    P_C = PENALIZACION_CHOQUE * conteo_choques

    bloques_giro = _calcular_bloques_giro(trayectoria)
    P_R = sum(penalizacion_bloque_giro(b) for b in bloques_giro)

    conteo_acciones_post_meta = _calcular_acciones_post_meta(trayectoria, llegada)
    P_A = PENALIZACION_ACCION_POST_META * conteo_acciones_post_meta

    conteo_q_prematuro = _calcular_detencion_prematura(cromosoma, es_valido)
    P_prem = PENALIZACION_DETENCION_PREMATURA_POR_Q * conteo_q_prematuro

    P_inv = 0 if es_valido else PENALIZACION_GLOBAL_INVALIDA

    J = D + tau + P_Q + P_C + P_R + P_A + P_prem + P_inv
    phi = -J

    return ResultadoEvaluacion(
        cromosoma=cromosoma,
        n=n,
        posicion_final=resultado_sim.posicion_final,
        llegada=llegada,
        D=D,
        llegadas=llegadas,
        l=valor_l,
        tau=tau,
        es_valido=es_valido,
        rho=rho,
        conteo_Q_intermedio=conteo_q_intermedio,
        conteo_choques=conteo_choques,
        bloques_giro=bloques_giro,
        conteo_acciones_post_meta=conteo_acciones_post_meta,
        conteo_Q_prematuro=conteo_q_prematuro,
        P_Q=P_Q,
        P_C=P_C,
        P_R=P_R,
        P_A=P_A,
        P_prem=P_prem,
        P_inv=P_inv,
        J=J,
        phi=phi,
        trayectoria=trayectoria,
    )