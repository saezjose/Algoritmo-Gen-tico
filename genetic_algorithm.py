"""
Módulo 4: Algoritmo Genético
==============================

Orquesta el ciclo evolutivo completo durante G generaciones:

1. Ranking lexicográfico por prioridad de factibilidad (rho, J, D, tau).
2. Elitismo obligatorio del mejor cromosoma global histórico.
3. Selección por ranking geométrico para los N-1 cupos restantes.
4. Cruzamiento de un punto.
5. Mutación independiente por gen.
6. Reevaluación completa de cada descendiente.

Parámetros configurables externamente: n, pm, N, G, ps, seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from maze_loader import Maze
from simulator import ACTIONS, Chromosome, simulate_chromosome
from evaluator import EvaluationResult, evaluate_chromosome


class GAParameterError(ValueError):
    """Error de validación de los parámetros del algoritmo genético."""


def validate_parameters(n: int, pm: float, N: int, G: int, ps: float) -> None:
    if n < 1:
        raise GAParameterError("La longitud del cromosoma (n) debe ser un entero positivo.")
    if not (0.0 <= pm <= 1.0):
        raise GAParameterError("La probabilidad de mutación (pm) debe estar en el rango [0, 1].")
    if N < 3 or N % 2 == 0:
        raise GAParameterError(
            f"El tamaño de población (N={N}) debe ser un entero impar mayor o igual a 3."
        )
    if G < 1:
        raise GAParameterError("El número de generaciones (G) debe ser un entero positivo.")
    if not (0.0 < ps < 1.0):
        raise GAParameterError("La presión selectiva (ps) debe estar estrictamente en (0, 1).")


def evaluate(chromosome: Chromosome, maze: Maze) -> EvaluationResult:
    """Simula y evalúa un cromosoma sobre el laberinto dado (evaluación completa)."""
    sim_result = simulate_chromosome(chromosome, maze)
    return evaluate_chromosome(chromosome, sim_result, maze)


def random_chromosome(n: int, rng: random.Random) -> Chromosome:
    """Genera un cromosoma aleatorio de longitud n muestreando uniformemente de A."""
    return tuple(rng.choice(ACTIONS) for _ in range(n))


def _cumulative_probabilities(N: int, ps: float) -> List[float]:
    """C_i = (1 - (1-ps)^i) / (1 - (1-ps)^N), para i = 1, ..., N."""
    denom = 1.0 - (1.0 - ps) ** N
    return [(1.0 - (1.0 - ps) ** i) / denom for i in range(1, N + 1)]


def select_parent(
    ranked_population: List[Chromosome], cumulative_probs: List[float], rng: random.Random
) -> Chromosome:
    """Selecciona un padre: i = min{j : u <= C_j}, con u ~ U(0,1)."""
    u = rng.random()
    for chromosome, c_i in zip(ranked_population, cumulative_probs):
        if u <= c_i:
            return chromosome
    return ranked_population[-1]  # salvaguarda numérica ante redondeo de punto flotante


def one_point_crossover(
    parent1: Chromosome, parent2: Chromosome, rng: random.Random
) -> Tuple[Chromosome, Chromosome]:
    """Cruzamiento de un punto: c en {1, ..., n-1}.

    x' = (g_1,...,g_c, h_{c+1},...,h_n)
    y' = (h_1,...,h_c, g_{c+1},...,g_n)
    """
    n = len(parent1)
    c = rng.randint(1, n - 1)
    child1 = parent1[:c] + parent2[c:]
    child2 = parent2[:c] + parent1[c:]
    return child1, child2


def mutate(chromosome: Chromosome, pm: float, rng: random.Random) -> Chromosome:
    """Mutación independiente por gen: cada gen muta con probabilidad pm,
    reemplazándose de forma uniforme por una de las otras 3 acciones."""
    genes = list(chromosome)
    for k in range(len(genes)):
        if rng.random() < pm:
            alternatives = [a for a in ACTIONS if a != genes[k]]
            genes[k] = rng.choice(alternatives)
    return tuple(genes)


@dataclass
class GAResult:
    """Resultado completo de una corrida del algoritmo genético."""

    maze: Maze
    n: int
    pm: float
    N: int
    G: int
    ps: float
    seed: int

    best_chromosome: Chromosome
    best_evaluation: EvaluationResult

    history_best_J: List[float] = field(default_factory=list)
    history_valid_proportion: List[float] = field(default_factory=list)
    all_evaluated: Dict[Chromosome, EvaluationResult] = field(default_factory=dict)

    def best_unique_chromosomes(self) -> Dict[Chromosome, EvaluationResult]:
        """X* = {x en U : J(x) = J*}: cromosomas únicos que empatan en el mejor J."""
        j_star = self.best_evaluation.J
        return {c: ev for c, ev in self.all_evaluated.items() if ev.J == j_star}


def run_genetic_algorithm(
    maze: Maze, n: int, pm: float, N: int, G: int, ps: float, seed: int
) -> GAResult:
    """Ejecuta el ciclo evolutivo completo y retorna un GAResult con historial y auditoría.

    Convención de generaciones: la generación 1 corresponde a la población
    inicial aleatoria (ya evaluada y clasificada); las generaciones 2..G se
    obtienen aplicando elitismo, selección, cruzamiento y mutación sobre la
    generación anterior. Se registran exactamente G puntos de historial.
    """
    validate_parameters(n, pm, N, G, ps)
    rng = random.Random(seed)

    population: List[Chromosome] = [random_chromosome(n, rng) for _ in range(N)]
    evaluations: List[EvaluationResult] = [evaluate(c, maze) for c in population]

    global_best_chromosome: Optional[Chromosome] = None
    global_best_evaluation: Optional[EvaluationResult] = None

    history_best_J: List[float] = []
    history_valid_proportion: List[float] = []
    all_evaluated: Dict[Chromosome, EvaluationResult] = {}

    n_offspring_needed = N - 1  # siempre par, dado que N es impar (validado arriba)

    for generation in range(1, G + 1):
        # Registrar todos los cromosomas de esta generación en el conjunto U.
        for chromosome, ev in zip(population, evaluations):
            all_evaluated.setdefault(chromosome, ev)

        # 1) Ranking lexicográfico (rho, J, D, tau): de mejor a peor.
        order = sorted(range(N), key=lambda idx: evaluations[idx].ranking_key())
        ranked_population = [population[i] for i in order]
        ranked_evaluations = [evaluations[i] for i in order]

        # 2) Actualización del mejor cromosoma global histórico.
        current_best_chromosome = ranked_population[0]
        current_best_evaluation = ranked_evaluations[0]
        if (
            global_best_evaluation is None
            or current_best_evaluation.ranking_key() < global_best_evaluation.ranking_key()
        ):
            global_best_chromosome = current_best_chromosome
            global_best_evaluation = current_best_evaluation

        history_best_J.append(global_best_evaluation.J)
        valid_count = sum(1 for ev in evaluations if ev.is_valid)
        history_valid_proportion.append(valid_count / N)

        if generation == G:
            break  # última generación: no es necesario generar descendencia adicional

        # 3) Selección por ranking geométrico + 4) Cruzamiento + 5) Mutación.
        cumulative_probs = _cumulative_probabilities(N, ps)
        offspring: List[Chromosome] = []
        for _ in range(n_offspring_needed // 2):
            parent1 = select_parent(ranked_population, cumulative_probs, rng)
            parent2 = select_parent(ranked_population, cumulative_probs, rng)
            child1, child2 = one_point_crossover(parent1, parent2, rng)
            child1 = mutate(child1, pm, rng)
            child2 = mutate(child2, pm, rng)
            offspring.append(child1)
            offspring.append(child2)

        # Elitismo obligatorio: el mejor global pasa sin modificaciones (no se reevalúa,
        # ya que su evaluación es determinista y no ha cambiado).
        next_population = [global_best_chromosome] + offspring
        next_evaluations = [global_best_evaluation] + [
            evaluate(c, maze) for c in offspring  # 6) Reevaluación completa por descendiente.
        ]

        population, evaluations = next_population, next_evaluations

    assert global_best_chromosome is not None and global_best_evaluation is not None

    return GAResult(
        maze=maze,
        n=n,
        pm=pm,
        N=N,
        G=G,
        ps=ps,
        seed=seed,
        best_chromosome=global_best_chromosome,
        best_evaluation=global_best_evaluation,
        history_best_J=history_best_J,
        history_valid_proportion=history_valid_proportion,
        all_evaluated=all_evaluated,
    )