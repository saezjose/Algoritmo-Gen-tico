from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from cargador_laberinto import Laberinto
from simulador import ACCIONES, Cromosoma, simular_cromosoma
from evaluador import ResultadoEvaluacion, evaluar_cromosoma


class ErrorParametroAG(ValueError):
    """Error de validación de los parámetros del algoritmo genético."""


def validar_parametros(n: int, pm: float, N: int, G: int, ps: float) -> None:
    if n < 1:
        raise ErrorParametroAG("La longitud del cromosoma (n) debe ser un entero positivo.")
    if not (0.0 <= pm <= 1.0):
        raise ErrorParametroAG("La probabilidad de mutación (pm) debe estar en el rango [0, 1].")
    if N < 3 or N % 2 == 0:
        raise ErrorParametroAG(
            f"El tamaño de población (N={N}) debe ser un entero impar mayor o igual a 3."
        )
    if G < 1:
        raise ErrorParametroAG("El número de generaciones (G) debe ser un entero positivo.")
    if not (0.0 < ps < 1.0):
        raise ErrorParametroAG("La presión selectiva (ps) debe estar estrictamente en (0, 1).")


def evaluar(cromosoma: Cromosoma, laberinto: Laberinto) -> ResultadoEvaluacion:
    resultado_sim = simular_cromosoma(cromosoma, laberinto)
    return evaluar_cromosoma(cromosoma, resultado_sim, laberinto)


def cromosoma_aleatorio(n: int, rng: random.Random) -> Cromosoma:
    return tuple(rng.choice(ACCIONES) for _ in range(n))


def _probabilidades_acumuladas(N: int, ps: float) -> List[float]:
    denominador = 1.0 - (1.0 - ps) ** N
    return [(1.0 - (1.0 - ps) ** i) / denominador for i in range(1, N + 1)]


def seleccionar_padre(
    poblacion_ordenada: List[Cromosoma], probs_acumuladas: List[float], rng: random.Random
) -> Cromosoma:
    u = rng.random()
    for cromosoma, c_i in zip(poblacion_ordenada, probs_acumuladas):
        if u <= c_i:
            return cromosoma
    return poblacion_ordenada[-1]  

def cruce_un_punto(
    padre1: Cromosoma, padre2: Cromosoma, rng: random.Random
) -> Tuple[Cromosoma, Cromosoma]:
    n = len(padre1)
    c = rng.randint(1, n - 1)
    hijo1 = padre1[:c] + padre2[c:]
    hijo2 = padre2[:c] + padre1[c:]
    return hijo1, hijo2


def mutar(cromosoma: Cromosoma, pm: float, rng: random.Random) -> Cromosoma:
    genes = list(cromosoma)
    for k in range(len(genes)):
        if rng.random() < pm:
            alternativas = [a for a in ACCIONES if a != genes[k]]
            genes[k] = rng.choice(alternativas)
    return tuple(genes)


@dataclass
class ResultadoAG:
    laberinto: Laberinto
    n: int
    pm: float
    N: int
    G: int
    ps: float
    semilla: int

    mejor_cromosoma: Cromosoma
    mejor_evaluacion: ResultadoEvaluacion

    historial_mejor_J: List[float] = field(default_factory=list)
    historial_proporcion_valida: List[float] = field(default_factory=list)
    todos_evaluados: Dict[Cromosoma, ResultadoEvaluacion] = field(default_factory=dict)

    def cromosomas_unicos_mejores(self) -> Dict[Cromosoma, ResultadoEvaluacion]:
        j_estrella = self.mejor_evaluacion.J
        return {c: ev for c, ev in self.todos_evaluados.items() if ev.J == j_estrella}


def ejecutar_algoritmo_genetico(
    laberinto: Laberinto, n: int, pm: float, N: int, G: int, ps: float, semilla: int
) -> ResultadoAG:
    validar_parametros(n, pm, N, G, ps)
    rng = random.Random(semilla)

    poblacion: List[Cromosoma] = [cromosoma_aleatorio(n, rng) for _ in range(N)]
    evaluaciones: List[ResultadoEvaluacion] = [evaluar(c, laberinto) for c in poblacion]

    mejor_cromosoma_global: Optional[Cromosoma] = None
    mejor_evaluacion_global: Optional[ResultadoEvaluacion] = None

    historial_mejor_J: List[float] = []
    historial_proporcion_valida: List[float] = []
    todos_evaluados: Dict[Cromosoma, ResultadoEvaluacion] = {}

    num_descendientes_necesarios = N - 1 

    for generacion in range(1, G + 1):
        
        for cromosoma, ev in zip(poblacion, evaluaciones):
            todos_evaluados.setdefault(cromosoma, ev)

        orden = sorted(range(N), key=lambda idx: evaluaciones[idx].clave_ranking())
        poblacion_ordenada = [poblacion[i] for i in orden]
        evaluaciones_ordenadas = [evaluaciones[i] for i in orden]

        mejor_cromosoma_actual = poblacion_ordenada[0]
        mejor_evaluacion_actual = evaluaciones_ordenadas[0]
        if (
            mejor_evaluacion_global is None
            or mejor_evaluacion_actual.clave_ranking() < mejor_evaluacion_global.clave_ranking()
        ):
            mejor_cromosoma_global = mejor_cromosoma_actual
            mejor_evaluacion_global = mejor_evaluacion_actual

        historial_mejor_J.append(mejor_evaluacion_global.J)
        conteo_validos = sum(1 for ev in evaluaciones if ev.es_valido)
        historial_proporcion_valida.append(conteo_validos / N)

        if generacion == G:
            break  
        probs_acumuladas = _probabilidades_acumuladas(N, ps)
        descendencia: List[Cromosoma] = []
        for _ in range(num_descendientes_necesarios // 2):
            padre1 = seleccionar_padre(poblacion_ordenada, probs_acumuladas, rng)
            padre2 = seleccionar_padre(poblacion_ordenada, probs_acumuladas, rng)
            hijo1, hijo2 = cruce_un_punto(padre1, padre2, rng)
            hijo1 = mutar(hijo1, pm, rng)
            hijo2 = mutar(hijo2, pm, rng)
            descendencia.append(hijo1)
            descendencia.append(hijo2)

        poblacion_siguiente = [mejor_cromosoma_global] + descendencia
        evaluaciones_siguiente = [mejor_evaluacion_global] + [
            evaluar(c, laberinto) for c in descendencia  
        ]

        poblacion, evaluaciones = poblacion_siguiente, evaluaciones_siguiente

    assert mejor_cromosoma_global is not None and mejor_evaluacion_global is not None

    return ResultadoAG(
        laberinto=laberinto,
        n=n,
        pm=pm,
        N=N,
        G=G,
        ps=ps,
        semilla=semilla,
        mejor_cromosoma=mejor_cromosoma_global,
        mejor_evaluacion=mejor_evaluacion_global,
        historial_mejor_J=historial_mejor_J,
        historial_proporcion_valida=historial_proporcion_valida,
        todos_evaluados=todos_evaluados,
    )