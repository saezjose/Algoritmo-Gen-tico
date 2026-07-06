"""
Punto de Entrada Principal
=============================
Algoritmo Genético para la Resolución de Laberintos — INFO-1159
Universidad Católica de Temuco

Uso:
    python main.py --csv laberinto.csv --n 40 --pm 0.15 --pop 51 \
                    --gen 200 --ps 0.2 --seed 42

Todos los parámetros son recibidos externamente, sin valores fijados
rígidamente en la lógica principal.
"""
from __future__ import annotations

import argparse
import sys

from maze_loader import load_maze_csv, MazeValidationError
from genetic_algorithm import run_genetic_algorithm, GAParameterError
from visualization import generate_full_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Algoritmo Genético para la resolución de laberintos (INFO-1159)."
    )
    parser.add_argument("--csv", required=True, help="Ruta del archivo CSV del laberinto.")
    parser.add_argument("--n", type=int, required=True, help="Longitud del cromosoma.")
    parser.add_argument(
        "--pm", type=float, required=True, help="Probabilidad de mutación por gen (0 a 1)."
    )
    parser.add_argument(
        "--pop", type=int, required=True, dest="N",
        help="Tamaño de la población (entero impar >= 3).",
    )
    parser.add_argument(
        "--gen", type=int, required=True, dest="G", help="Número total de generaciones."
    )
    parser.add_argument(
        "--ps", type=float, required=True, help="Presión selectiva (0 a 1, exclusivo)."
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Semilla aleatoria para reproducibilidad."
    )
    parser.add_argument(
        "--output-dir", type=str, default="resultados",
        help="Directorio donde se guardarán las gráficas generadas (default: 'resultados').",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="No desplegar las gráficas interactivamente (solo guardarlas en disco).",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        maze = load_maze_csv(args.csv)
    except (FileNotFoundError, MazeValidationError) as exc:
        print(f"ERROR al cargar el laberinto: {exc}", file=sys.stderr)
        return 1

    try:
        ga_result = run_genetic_algorithm(
            maze=maze, n=args.n, pm=args.pm, N=args.N, G=args.G, ps=args.ps, seed=args.seed,
        )
    except GAParameterError as exc:
        print(f"ERROR en los parámetros del algoritmo genético: {exc}", file=sys.stderr)
        return 1

    generate_full_report(ga_result, output_dir=args.output_dir, show_plots=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())