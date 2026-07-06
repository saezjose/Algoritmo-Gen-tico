"""
Módulo 5: Resultados y Visualización
=======================================

Genera y despliega, usando Matplotlib:

  - Gráfica 1: mejor valor global de J(x) por generación, en escala logarítmica.
  - Gráfica 2: proporción de soluciones válidas en la población por generación.
  - Consola/Texto: cromosomas únicos con el mejor valor objetivo y sus pasos.
  - Consola/Auditoría: trayectoria paso a paso del/los mejor(es) cromosoma(s).

Convención de coordenadas para la auditoría: se reportan en formato (X, Y),
donde X es la columna del mapa e Y es la fila, ambas en notación 1-indexada
para ser consistentes con la notación matemática (i, j) de la pauta.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt

from evaluator import EvaluationResult
from genetic_algorithm import Chromosome, GAResult
from maze_loader import Maze


def plot_best_objective(ga_result: GAResult, output_dir: str, show: bool = True) -> str:
    """Gráfica 1: mejor J(x) global por generación, en escala logarítmica."""
    fig, ax = plt.subplots(figsize=(8, 5))
    generations = range(1, len(ga_result.history_best_J) + 1)
    ax.semilogy(
        generations, ga_result.history_best_J,
        marker="o", markersize=3, linewidth=1.2, color="#1f77b4",
    )
    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor J(x) global (escala logarítmica)")
    ax.set_title("Evolución del mejor valor de la función objetivo")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(output_dir, "grafica_1_mejor_objetivo.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_valid_proportion(ga_result: GAResult, output_dir: str, show: bool = True) -> str:
    """Gráfica 2: proporción de soluciones válidas en la población por generación."""
    fig, ax = plt.subplots(figsize=(8, 5))
    generations = range(1, len(ga_result.history_valid_proportion) + 1)
    ax.plot(
        generations, ga_result.history_valid_proportion,
        marker="o", markersize=3, linewidth=1.2, color="#2ca02c",
    )
    ax.set_xlabel("Generación")
    ax.set_ylabel("Proporción de soluciones válidas")
    ax.set_title("Proporción de soluciones válidas en la población por generación")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(output_dir, "grafica_2_proporcion_validas.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _describe_steps(evaluation: EvaluationResult) -> str:
    if evaluation.is_valid:
        return f"{evaluation.tau} pasos hasta la llegada válida (cromosoma de {evaluation.n} genes)"
    return f"no alcanza la meta de forma válida (cromosoma completo de {evaluation.n} genes)"


def print_best_chromosomes_report(ga_result: GAResult) -> None:
    """Consola/Texto: lista detallada de cromosomas únicos con el mejor J(x)."""
    best_unique = ga_result.best_unique_chromosomes()

    print("=" * 78)
    print("CONSOLA / TEXTO — Cromosomas únicos con el mejor valor objetivo")
    print("=" * 78)
    print(f"Mejor valor de función objetivo encontrado: J* = {ga_result.best_evaluation.J}")
    print(f"Fitness asociado:                          phi* = {ga_result.best_evaluation.phi}")
    print(f"Cantidad de cromosomas únicos que alcanzan J*:  {len(best_unique)}")
    print(f"Solución válida:                            {ga_result.best_evaluation.is_valid}")
    print("-" * 78)
    for idx, (chromosome, evaluation) in enumerate(best_unique.items(), start=1):
        print(f"[{idx}] Cromosoma: {''.join(chromosome)}")
        print(f"      {_describe_steps(evaluation)}")
        print(
            f"      D(x)={evaluation.D}  tau(x)={evaluation.tau}  "
            f"choques={evaluation.collision_count}  "
            f"pausas_intermedias={evaluation.Q_intermediate_count}  "
            f"acciones_post_meta={evaluation.post_goal_active_count}  "
            f"bloques_giro={list(evaluation.turn_blocks)}"
        )
    print("=" * 78)


def print_audited_trajectory(chromosome: Chromosome, evaluation: EvaluationResult, maze: Maze) -> None:
    """Consola/Auditoría: trayectoria paso a paso en coordenadas (X, Y), 1-indexadas.

    X corresponde a la columna del mapa; Y corresponde a la fila.
    """
    print("-" * 78)
    print(f"CONSOLA / AUDITORÍA — trayectoria de {''.join(chromosome)}")
    print("-" * 78)

    start_i, start_j = maze.start
    print(f"Paso 0 (inicio): (X={start_j + 1}, Y={start_i + 1})  dirección=S")

    for rec in evaluation.trajectory:
        i, j = rec.position_after
        nota = "  [CHOQUE: movimiento bloqueado]" if (rec.gene == "M" and rec.collided) else ""
        print(
            f"Paso {rec.step_index:>3}: gen={rec.gene}  "
            f"(X={j + 1}, Y={i + 1})  dirección={rec.direction_after}{nota}"
        )
    print("-" * 78)


def generate_full_report(ga_result: GAResult, output_dir: Optional[str] = "resultados",
                          show_plots: bool = True) -> None:
    """Genera el conjunto completo de resultados mínimos exigidos por la pauta."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path1 = plot_best_objective(ga_result, output_dir, show=show_plots)
        path2 = plot_valid_proportion(ga_result, output_dir, show=show_plots)
        print(f"Gráfica 1 (mejor objetivo, escala log) guardada en: {path1}")
        print(f"Gráfica 2 (proporción de válidas) guardada en:      {path2}")
        print()

    print_best_chromosomes_report(ga_result)
    print()

    best_unique = ga_result.best_unique_chromosomes()
    for chromosome, evaluation in best_unique.items():
        print_audited_trajectory(chromosome, evaluation, ga_result.maze)
        print()