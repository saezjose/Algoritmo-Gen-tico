"""
Módulo 1: Lectura y Validación del Laberinto
=============================================

Responsable de leer un archivo CSV externo y transformarlo en una matriz
L ∈ {0, 1, 2, X}^(m x r), aplicando validación estricta según la pauta:

- Símbolos válidos: 0 (libre), 1 (salida), 2 (llegada), X (muro).
- Muros perimetrales obligatorios: toda la primera/última fila y columna
  deben ser muros.
- Unicidad de extremos: debe existir exactamente una salida y una llegada.
- Ubicación de extremos: la salida debe estar en la primera fila interior
  (fila 2 en notación 1-indexada del enunciado); la llegada en la última
  fila interior (fila m-1 en notación 1-indexada).
- Zona despejada (vecindario de Moore): las celdas interiores que rodean
  a la salida y a la llegada no pueden ser muros.

Nota de indexación: internamente se usa indexación 0-based de Python (como
es natural en el lenguaje), pero toda la documentación hace explícita la
correspondencia con la notación 1-indexada usada en el enunciado matemático
de la pauta, para evitar errores de "off-by-one".
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# --- Símbolos del alfabeto del laberinto ---
WALL = "X"
FREE = "0"
START = "1"
GOAL = "2"
VALID_SYMBOLS = {FREE, START, GOAL, WALL}

Position = Tuple[int, int]


class MazeValidationError(ValueError):
    """Error específico de validación estructural del laberinto."""


@dataclass(frozen=True)
class Maze:
    """Representación inmutable y validada de un laberinto.

    Atributos
    ---------
    grid: matriz L, 0-indexada, de tamaño n_rows x n_cols.
    n_rows: número de filas, m.
    n_cols: número de columnas, r.
    start: posición (i_s, j_s) de la salida, 0-indexada.
    goal: posición (i_z, j_z) de la llegada, 0-indexada.
    """

    grid: Tuple[Tuple[str, ...], ...]
    n_rows: int
    n_cols: int
    start: Position
    goal: Position

    def in_bounds(self, pos: Position) -> bool:
        """Verifica que una posición esté dentro de los límites de la matriz.

        El motor de ejecución debe invocar siempre esta función antes de
        indexar la matriz, para evitar errores de indexación al intentar
        salir del laberinto.
        """
        i, j = pos
        return 0 <= i < self.n_rows and 0 <= j < self.n_cols

    def is_wall(self, pos: Position) -> bool:
        i, j = pos
        return self.grid[i][j] == WALL

    def is_transitable(self, pos: Position) -> bool:
        """Una celda es transitable si está dentro de los límites y no es muro.

        Las celdas con símbolo 0, 1 o 2 se consideran transitables.
        """
        return self.in_bounds(pos) and not self.is_wall(pos)


def _read_raw_grid(path: str | Path) -> List[List[str]]:
    """Lee el archivo CSV y retorna una matriz de strings, validando forma rectangular."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo CSV del laberinto en la ruta indicada: {path}"
        )

    rows: List[List[str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            # Se descartan celdas vacías generadas por comas finales o espacios sobrantes.
            cleaned = [cell.strip() for cell in raw_row if cell.strip() != ""]
            if cleaned:
                rows.append(cleaned)

    if not rows:
        raise MazeValidationError("El archivo CSV está vacío o no contiene celdas válidas.")

    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise MazeValidationError(
            "El laberinto debe ser rectangular: todas las filas deben tener la misma "
            f"cantidad de columnas. Longitudes de fila encontradas: {sorted(lengths)}."
        )

    return rows


def _validate_symbols(grid: Tuple[Tuple[str, ...], ...]) -> None:
    invalid = [
        (i, j, cell)
        for i, row in enumerate(grid)
        for j, cell in enumerate(row)
        if cell not in VALID_SYMBOLS
    ]
    if invalid:
        raise MazeValidationError(
            f"Símbolos inválidos en el laberinto. Solo se permiten {sorted(VALID_SYMBOLS)}. "
            f"Celdas problemáticas (fila, columna, símbolo) [0-indexado]: {invalid}"
        )


def _validate_perimeter(grid: Tuple[Tuple[str, ...], ...], m: int, r: int) -> None:
    """Garantiza L[i,j] = X si i=1 v i=m v j=1 v j=r (en notación 1-indexada);
    equivalente 0-indexado: i=0 v i=m-1 v j=0 v j=r-1."""
    errors = [
        (i, j)
        for i in range(m)
        for j in range(r)
        if (i == 0 or i == m - 1 or j == 0 or j == r - 1) and grid[i][j] != WALL
    ]
    if errors:
        raise MazeValidationError(
            "El perímetro completo del laberinto debe estar compuesto por muros ('X'). "
            f"Celdas perimetrales inválidas (0-indexadas): {errors}"
        )


def _find_unique_symbol(
    grid: Tuple[Tuple[str, ...], ...], symbol: str, m: int, r: int, label: str
) -> Position:
    found = [(i, j) for i in range(m) for j in range(r) if grid[i][j] == symbol]
    if len(found) == 0:
        raise MazeValidationError(
            f"No se encontró ninguna celda con símbolo '{symbol}' ({label}). "
            "Debe existir exactamente una."
        )
    if len(found) > 1:
        raise MazeValidationError(
            f"Se encontraron {len(found)} celdas con símbolo '{symbol}' ({label}); "
            f"debe existir exactamente una. Posiciones (0-indexadas): {found}"
        )
    return found[0]


def _validate_extremes_location(start: Position, goal: Position, m: int) -> None:
    """Valida i_s = 2 y i_z = m-1 en notación 1-indexada del enunciado.

    Equivalente 0-indexado: fila de salida == 1, fila de llegada == m-2.
    """
    i_s, _ = start
    i_z, _ = goal

    if i_s != 1:
        raise MazeValidationError(
            "La salida debe ubicarse en la primera fila interior válida "
            f"(fila 2 en notación 1-indexada; índice 1 en 0-indexada). "
            f"Se encontró en la fila 0-indexada {i_s}."
        )
    if i_z != m - 2:
        raise MazeValidationError(
            "La llegada debe ubicarse en la última fila interior válida "
            f"(fila m-1 en notación 1-indexada; índice m-2 en 0-indexada). "
            f"Se encontró en la fila 0-indexada {i_z}, se esperaba {m - 2}."
        )


def _moore_neighbors(pos: Position) -> List[Position]:
    i, j = pos
    return [
        (i + di, j + dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
    ]


def _is_interior(pos: Position, m: int, r: int) -> bool:
    """Pertenece al conjunto interior I = {2,...,m-1} x {2,...,r-1} (1-indexado),
    equivalente 0-indexado: {1,...,m-2} x {1,...,r-2}."""
    i, j = pos
    return 1 <= i <= m - 2 and 1 <= j <= r - 2


def _validate_clear_zone(
    grid: Tuple[Tuple[str, ...], ...], pos: Position, m: int, r: int, label: str
) -> None:
    """Valida que N_8(pos) ∩ I no contenga muros, para pos en {salida, llegada}."""
    for neighbor in _moore_neighbors(pos):
        if _is_interior(neighbor, m, r) and grid[neighbor[0]][neighbor[1]] == WALL:
            raise MazeValidationError(
                f"La zona interior alrededor de {label} (posición 0-indexada {pos}) "
                f"contiene un muro en la celda vecina interior {neighbor}. "
                "El vecindario de Moore interior debe estar despejado."
            )


def load_maze_csv(path: str | Path) -> Maze:
    """Carga, valida y retorna un laberinto a partir de un archivo CSV.

    Lanza FileNotFoundError o MazeValidationError con mensajes descriptivos
    ante cualquier incumplimiento de las reglas estrictas de la pauta.
    """
    raw = _read_raw_grid(path)
    grid: Tuple[Tuple[str, ...], ...] = tuple(tuple(row) for row in raw)
    m = len(grid)
    r = len(grid[0])

    if m < 5 or r < 5:
        # Se requiere al menos 1 fila de perímetro + 1 fila interior a cada lado
        # más separación entre salida y llegada; 5x5 es el mínimo estructuralmente
        # razonable (perímetro + salida + al menos una fila intermedia + llegada + perímetro).
        raise MazeValidationError(
            f"El laberinto debe tener al menos 5 filas y 5 columnas para admitir "
            f"perímetro, salida y llegada en filas interiores distintas. Se recibió {m}x{r}."
        )

    _validate_symbols(grid)
    _validate_perimeter(grid, m, r)

    start = _find_unique_symbol(grid, START, m, r, "salida")
    goal = _find_unique_symbol(grid, GOAL, m, r, "llegada")

    _validate_extremes_location(start, goal, m)

    _validate_clear_zone(grid, start, m, r, "la salida")
    _validate_clear_zone(grid, goal, m, r, "la llegada")

    return Maze(grid=grid, n_rows=m, n_cols=r, start=start, goal=goal)