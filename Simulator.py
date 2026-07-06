"""
Módulo 2: Motor de Ejecución (Simulador)
==========================================

Ejecuta secuencialmente un cromosoma x = (g_1, ..., g_n) sobre un laberinto,
partiendo desde la posición de salida mirando hacia el Sur, aplicando las
reglas de giro, avance y quietud definidas en la pauta.

Alfabeto de acciones: A = {H, A, M, Q}
Direcciones:          D = {N, E, S, O}
Vectores de desplazamiento:
    v_N = (-1, 0), v_E = (0, 1), v_S = (1, 0), v_O = (0, -1)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from maze_loader import Maze, Position

Chromosome = Tuple[str, ...]

ACTIONS: Tuple[str, ...] = ("H", "A", "M", "Q")
DIRECTIONS: Tuple[str, ...] = ("N", "E", "S", "O")
DIRECTION_VECTORS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "O": (0, -1),
}
INITIAL_DIRECTION = "S"


@dataclass(frozen=True)
class StepRecord:
    """Registro auditable de un paso t de ejecución del cromosoma."""

    step_index: int  # t, 1-indexado (para reportes)
    gene: str
    position_before: Position  # p_{t-1}
    position_after: Position  # p_t
    direction_before: str  # d_{t-1}
    direction_after: str  # d_t
    collided: bool  # True si el gen fue M y el intento de avance falló


@dataclass(frozen=True)
class SimulationResult:
    trajectory: Tuple[StepRecord, ...]
    final_position: Position
    final_direction: str


def _turn_clockwise(direction: str) -> str:
    idx = DIRECTIONS.index(direction)
    return DIRECTIONS[(idx + 1) % 4]


def _turn_counterclockwise(direction: str) -> str:
    idx = DIRECTIONS.index(direction)
    return DIRECTIONS[(idx - 1) % 4]


def simulate_chromosome(chromosome: Chromosome, maze: Maze) -> SimulationResult:
    """Simula la ejecución completa de un cromosoma sobre el laberinto dado.

    p_0 = s (salida), d_0 = S (sur). Para cada gen g_t:
      - H: gira 90° horario, posición no cambia.
      - A: gira 90° antihorario, posición no cambia.
      - Q: no cambia posición ni dirección.
      - M: calcula p~_t = p_{t-1} + v_{d_{t-1}}; si es transitable y está
           dentro de límites, avanza (p_t = p~_t); si no, p_t = p_{t-1}
           y se registra un choque.
    """
    position: Position = maze.start
    direction = INITIAL_DIRECTION
    trajectory: List[StepRecord] = []

    for t, gene in enumerate(chromosome, start=1):
        pos_before = position
        dir_before = direction
        collided = False

        if gene == "H":
            direction = _turn_clockwise(direction)
        elif gene == "A":
            direction = _turn_counterclockwise(direction)
        elif gene == "Q":
            pass
        elif gene == "M":
            dr, dc = DIRECTION_VECTORS[direction]
            tentative: Position = (position[0] + dr, position[1] + dc)
            if maze.is_transitable(tentative):
                position = tentative
            else:
                collided = True
        else:
            raise ValueError(
                f"Gen inválido detectado en el cromosoma: {gene!r}. "
                f"Se esperaba uno de {ACTIONS}."
            )

        trajectory.append(
            StepRecord(
                step_index=t,
                gene=gene,
                position_before=pos_before,
                position_after=position,
                direction_before=dir_before,
                direction_after=direction,
                collided=collided,
            )
        )

    return SimulationResult(
        trajectory=tuple(trajectory),
        final_position=position,
        final_direction=direction,
    )