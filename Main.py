"""
python main.py --csv laberinto.csv --n 40 --pm 0.15 --pop 51 --gen 200 --ps 0.2 --seed 42
"""
from __future__ import annotations
import argparse
import sys
from cargador_laberinto import cargar_laberinto_csv, ErrorValidacionLaberinto
from algoritmo_genetico import ejecutar_algoritmo_genetico, ErrorParametroAG
from visualizacion import generar_reporte_completo


def construir_parser_argumentos() -> argparse.ArgumentParser:
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
        "--seed", type=int, required=True, dest="semilla",
        help="Semilla aleatoria para reproducibilidad.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="resultados", dest="directorio_salida",
        help="Directorio donde se guardarán las gráficas generadas (default: 'resultados').",
    )
    parser.add_argument(
        "--no-show", action="store_true", dest="no_mostrar",
        help="No desplegar las gráficas interactivamente (solo guardarlas en disco).",
    )
    return parser


def principal() -> int:
    parser = construir_parser_argumentos()
    args = parser.parse_args()

    try:
        laberinto = cargar_laberinto_csv(args.csv)
    except (FileNotFoundError, ErrorValidacionLaberinto) as exc:
        print(f"ERROR al cargar el laberinto: {exc}", file=sys.stderr)
        return 1

    try:
        resultado_ag = ejecutar_algoritmo_genetico(
            laberinto=laberinto, n=args.n, pm=args.pm, N=args.N, G=args.G,
            ps=args.ps, semilla=args.semilla,
        )
    except ErrorParametroAG as exc:
        print(f"ERROR en los parámetros del algoritmo genético: {exc}", file=sys.stderr)
        return 1

    generar_reporte_completo(
        resultado_ag, directorio_salida=args.directorio_salida, mostrar_graficas=not args.no_mostrar
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())