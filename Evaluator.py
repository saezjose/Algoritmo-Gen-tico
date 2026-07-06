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
   (PENALTY_GLOBAL_INVALID) fácilmente ajustable si la cátedra confirma
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

from maze_loader import Maze, Position
from simulator import Chromosome, SimulationResult, StepRecord

# --- Constantes de penalización (parametrizadas como constantes con nombre) ---
PENALTY_INTERMEDIATE_PAUSE = 10       # P_Q: por cada Q intermedio
PENALTY_COLLISION = 30                # P_C: por cada choque (M fallido)
PENALTY_POST_GOAL_ACTIVE = 100        # P_A: por cada acción activa tras la meta
PENALTY_PREMATURE_STOP_PER_Q = 10     # P_prem: por cada Q de la cola final (si inválido)
PENALTY_GLOBAL_INVALID = 10_000       # P_inv: penalización global si no es solución válida


def turn_block_penalty(b: int) -> int:
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
class EvaluationResult:
    """Resultado completo de evaluar un cromosoma sobre un laberinto."""

    chromosome: Chromosome
    n: int
    final_position: Position
    goal: Position

    D: int                      # Distancia de Manhattan final
    arrivals: Tuple[int, ...]   # T_z(x): pasos (1-indexados) de llegada efectiva
    l: int | None               # ℓ(x): última llegada efectiva (o None)
    tau: int                    # τ(x)
    is_valid: bool              # Solución válida completa
    rho: int                    # 0, 1 o 2 (prioridad de factibilidad)

    Q_intermediate_count: int
    collision_count: int
    turn_blocks: Tuple[int, ...]
    post_goal_active_count: int
    Q_premature_count: int

    P_Q: float
    P_C: float
    P_R: float
    P_A: float
    P_prem: float
    P_inv: float

    J: float
    phi: float

    trajectory: Tuple[StepRecord, ...]

    def ranking_key(self) -> Tuple[int, float, int, int]:
        """Clave lexicográfica (rho(x), J(x), D(x), tau(x)) para el ranking evolutivo."""
        return (self.rho, self.J, self.D, self.tau)


def _compute_manhattan_distance(final_position: Position, goal: Position) -> int:
    return abs(final_position[0] - goal[0]) + abs(final_position[1] - goal[1])


def _compute_arrivals(trajectory: Tuple[StepRecord, ...], goal: Position) -> Tuple[int, ...]:
    """T_z(x) = {t : p_{t-1} != z ^ p_t = z}."""
    return tuple(
        rec.step_index
        for rec in trajectory
        if rec.position_before != goal and rec.position_after == goal
    )


def _compute_tau(
    arrivals: Tuple[int, ...], trajectory: Tuple[StepRecord, ...], n: int
) -> Tuple[int, bool, int | None]:
    """Calcula (tau(x), es_valido, ell(x))."""
    if not arrivals:
        return n + 1, False, None

    l_val = max(arrivals)
    if l_val < n and all(trajectory[k].gene == "Q" for k in range(l_val, n)):
        return l_val, True, l_val
    return n + 1, False, l_val


def _compute_intermediate_pauses(chromosome: Chromosome) -> int:
    """Q_int(x) = #{k in {1,...,n-1} : g_k = Q ^ exists j>k con g_j != Q}."""
    n = len(chromosome)
    last_non_q_idx: int | None = None
    for idx in range(n - 1, -1, -1):
        if chromosome[idx] != "Q":
            last_non_q_idx = idx
            break
    if last_non_q_idx is None:
        return 0  # todos los genes son Q: no existe "parte activa" posterior
    return sum(1 for idx in range(0, last_non_q_idx) if chromosome[idx] == "Q")


def _compute_collisions(trajectory: Tuple[StepRecord, ...]) -> int:
    return sum(1 for rec in trajectory if rec.collided)


def _compute_turn_blocks(trajectory: Tuple[StepRecord, ...]) -> Tuple[int, ...]:
    """Construye el multiconjunto B(x) de longitudes de bloques de giros.

    Un bloque se compone de acciones H/A ejecutadas en la misma celda.
    Q y M fallido no modifican el contador del bloque. Un M exitoso cierra
    el bloque (lo agrega a B(x)) y reinicia el contador a 0. El bloque
    remanente al finalizar el cromosoma también se cierra (ver docstring
    del módulo).
    """
    blocks: List[int] = []
    current_block = 0
    for rec in trajectory:
        if rec.gene in ("H", "A"):
            current_block += 1
        elif rec.gene == "M" and not rec.collided:
            blocks.append(current_block)
            current_block = 0
        # Q y M fallido: no modifican current_block
    if current_block > 0:
        blocks.append(current_block)
    return tuple(blocks)


def _compute_post_goal_active(trajectory: Tuple[StepRecord, ...], goal: Position) -> int:
    """A_meta(x): acciones activas (H, A, M) ejecutadas ya estando en la meta."""
    return sum(
        1
        for rec in trajectory
        if rec.position_before == goal and rec.gene in ("H", "A", "M")
    )


def _compute_premature_stop(chromosome: Chromosome, is_valid: bool) -> int:
    """Q_prem(x): longitud de la cola final de genes Q, solo si la solución es inválida."""
    if is_valid:
        return 0
    count = 0
    for gene in reversed(chromosome):
        if gene == "Q":
            count += 1
        else:
            break
    return count


def evaluate_chromosome(
    chromosome: Chromosome, sim_result: SimulationResult, maze: Maze
) -> EvaluationResult:
    """Evalúa completamente un cromosoma ya simulado, calculando J(x) y phi(x)."""
    n = len(chromosome)
    goal = maze.goal
    trajectory = sim_result.trajectory

    D = _compute_manhattan_distance(sim_result.final_position, goal)
    arrivals = _compute_arrivals(trajectory, goal)
    tau, is_valid, l_val = _compute_tau(arrivals, trajectory, n)

    if is_valid:
        rho = 0
    elif arrivals:
        rho = 1
    else:
        rho = 2

    q_intermediate_count = _compute_intermediate_pauses(chromosome)
    P_Q = PENALTY_INTERMEDIATE_PAUSE * q_intermediate_count

    collision_count = _compute_collisions(trajectory)
    P_C = PENALTY_COLLISION * collision_count

    turn_blocks = _compute_turn_blocks(trajectory)
    P_R = sum(turn_block_penalty(b) for b in turn_blocks)

    post_goal_active_count = _compute_post_goal_active(trajectory, goal)
    P_A = PENALTY_POST_GOAL_ACTIVE * post_goal_active_count

    q_premature_count = _compute_premature_stop(chromosome, is_valid)
    P_prem = PENALTY_PREMATURE_STOP_PER_Q * q_premature_count

    P_inv = 0 if is_valid else PENALTY_GLOBAL_INVALID

    J = D + tau + P_Q + P_C + P_R + P_A + P_prem + P_inv
    phi = -J

    return EvaluationResult(
        chromosome=chromosome,
        n=n,
        final_position=sim_result.final_position,
        goal=goal,
        D=D,
        arrivals=arrivals,
        l=l_val,
        tau=tau,
        is_valid=is_valid,
        rho=rho,
        Q_intermediate_count=q_intermediate_count,
        collision_count=collision_count,
        turn_blocks=turn_blocks,
        post_goal_active_count=post_goal_active_count,
        Q_premature_count=q_premature_count,
        P_Q=P_Q,
        P_C=P_C,
        P_R=P_R,
        P_A=P_A,
        P_prem=P_prem,
        P_inv=P_inv,
        J=J,
        phi=phi,
        trajectory=trajectory,
    )