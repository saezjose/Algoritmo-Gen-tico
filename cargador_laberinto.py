from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


MURO = "X"
LIBRE = "0"
SIMBOLO_SALIDA = "1"
SIMBOLO_LLEGADA = "2"
SIMBOLOS_VALIDOS = {LIBRE, SIMBOLO_SALIDA, SIMBOLO_LLEGADA, MURO}

Posicion = Tuple[int, int]

class ErrorValidacionLaberinto(ValueError):
    """Error específico de validación estructural del laberinto."""

@dataclass(frozen=True)
class Laberinto:
    cuadricula: Tuple[Tuple[str, ...], ...]
    num_filas: int
    num_columnas: int
    salida: Posicion
    llegada: Posicion

    def dentro_de_limites(self, pos: Posicion) -> bool:
        i, j = pos
        return 0 <= i < self.num_filas and 0 <= j < self.num_columnas

    def es_muro(self, pos: Posicion) -> bool:
        i, j = pos
        return self.cuadricula[i][j] == MURO

    def es_transitable(self, pos: Posicion) -> bool:
        return self.dentro_de_limites(pos) and not self.es_muro(pos)


def _leer_cuadricula_cruda(ruta: str | Path) -> List[List[str]]:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo CSV del laberinto en la ruta indicada: {ruta}"
        )

    filas: List[List[str]] = []
    with ruta.open(newline="", encoding="utf-8-sig") as f:
        lector = csv.reader(f)
        for fila_cruda in lector:
            # Se descartan celdas vacías generadas por comas finales o espacios sobrantes.
            limpia = [celda.strip() for celda in fila_cruda if celda.strip() != ""]
            if limpia:
                filas.append(limpia)

    if not filas:
        raise ErrorValidacionLaberinto("El archivo CSV está vacío o no contiene celdas válidas.")

    longitudes = {len(fila) for fila in filas}
    if len(longitudes) != 1:
        raise ErrorValidacionLaberinto(
            "El laberinto debe ser rectangular: todas las filas deben tener la misma "
            f"cantidad de columnas. Longitudes de fila encontradas: {sorted(longitudes)}."
        )

    return filas


def _validar_simbolos(cuadricula: Tuple[Tuple[str, ...], ...]) -> None:
    invalidas = [
        (i, j, celda)
        for i, fila in enumerate(cuadricula)
        for j, celda in enumerate(fila)
        if celda not in SIMBOLOS_VALIDOS
    ]
    if invalidas:
        raise ErrorValidacionLaberinto(
            f"Símbolos inválidos en el laberinto. Solo se permiten {sorted(SIMBOLOS_VALIDOS)}. "
            f"Celdas problemáticas (fila, columna, símbolo) [0-indexado]: {invalidas}"
        )


def _validar_perimetro(cuadricula: Tuple[Tuple[str, ...], ...], m: int, r: int) -> None:
    errores = [
        (i, j)
        for i in range(m)
        for j in range(r)
        if (i == 0 or i == m - 1 or j == 0 or j == r - 1) and cuadricula[i][j] != MURO
    ]
    if errores:
        raise ErrorValidacionLaberinto(
            "El perímetro completo del laberinto debe estar compuesto por muros ('X'). "
            f"Celdas perimetrales inválidas (0-indexadas): {errores}"
        )


def _encontrar_simbolo_unico(
    cuadricula: Tuple[Tuple[str, ...], ...], simbolo: str, m: int, r: int, etiqueta: str
) -> Posicion:
    encontrados = [(i, j) for i in range(m) for j in range(r) if cuadricula[i][j] == simbolo]
    if len(encontrados) == 0:
        raise ErrorValidacionLaberinto(
            f"No se encontró ninguna celda con símbolo '{simbolo}' ({etiqueta}). "
            "Debe existir exactamente una."
        )
    if len(encontrados) > 1:
        raise ErrorValidacionLaberinto(
            f"Se encontraron {len(encontrados)} celdas con símbolo '{simbolo}' ({etiqueta}); "
            f"debe existir exactamente una. Posiciones (0-indexadas): {encontrados}"
        )
    return encontrados[0]


def _validar_ubicacion_extremos(salida: Posicion, llegada: Posicion, m: int) -> None:
    i_s, _ = salida
    i_z, _ = llegada

    if i_s != 1:
        raise ErrorValidacionLaberinto(
            "La salida debe ubicarse en la primera fila interior válida "
            f"(fila 2 en notación 1-indexada; índice 1 en 0-indexada). "
            f"Se encontró en la fila 0-indexada {i_s}."
        )
    if i_z != m - 2:
        raise ErrorValidacionLaberinto(
            "La llegada debe ubicarse en la última fila interior válida "
            f"(fila m-1 en notación 1-indexada; índice m-2 en 0-indexada). "
            f"Se encontró en la fila 0-indexada {i_z}, se esperaba {m - 2}."
        )


def _vecinos_moore(pos: Posicion) -> List[Posicion]:
    i, j = pos
    return [
        (i + di, j + dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
    ]


def _es_interior(pos: Posicion, m: int, r: int) -> bool:
    i, j = pos
    return 1 <= i <= m - 2 and 1 <= j <= r - 2


def _validar_zona_despejada(
    cuadricula: Tuple[Tuple[str, ...], ...], pos: Posicion, m: int, r: int, etiqueta: str
) -> None:
    for vecino in _vecinos_moore(pos):
        if _es_interior(vecino, m, r) and cuadricula[vecino[0]][vecino[1]] == MURO:
            raise ErrorValidacionLaberinto(
                f"La zona interior alrededor de {etiqueta} (posición 0-indexada {pos}) "
                f"contiene un muro en la celda vecina interior {vecino}. "
                "El vecindario de Moore interior debe estar despejado."
            )


def cargar_laberinto_csv(ruta: str | Path) -> Laberinto:
    crudo = _leer_cuadricula_cruda(ruta)
    cuadricula: Tuple[Tuple[str, ...], ...] = tuple(tuple(fila) for fila in crudo)
    m = len(cuadricula)
    r = len(cuadricula[0])

    if m < 5 or r < 5:
        raise ErrorValidacionLaberinto(
            f"El laberinto debe tener al menos 5 filas y 5 columnas para admitir "
            f"perímetro, salida y llegada en filas interiores distintas. Se recibió {m}x{r}."
        )

    _validar_simbolos(cuadricula)
    _validar_perimetro(cuadricula, m, r)

    salida = _encontrar_simbolo_unico(cuadricula, SIMBOLO_SALIDA, m, r, "salida")
    llegada = _encontrar_simbolo_unico(cuadricula, SIMBOLO_LLEGADA, m, r, "llegada")

    _validar_ubicacion_extremos(salida, llegada, m)

    _validar_zona_despejada(cuadricula, salida, m, r, "la salida")
    _validar_zona_despejada(cuadricula, llegada, m, r, "la llegada")

    return Laberinto(
        cuadricula=cuadricula, num_filas=m, num_columnas=r, salida=salida, llegada=llegada
    )