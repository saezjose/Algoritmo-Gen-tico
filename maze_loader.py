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
    """Representación inmutable y validada de un laberinto.

    Atributos
    ---------
    cuadricula: matriz L, 0-indexada, de tamaño num_filas x num_columnas.
    num_filas: número de filas, m.
    num_columnas: número de columnas, r.
    salida: posición (i_s, j_s) de la salida, 0-indexada.
    llegada: posición (i_z, j_z) de la llegada, 0-indexada.
    """

    cuadricula: Tuple[Tuple[str, ...], ...]
    num_filas: int
    num_columnas: int
    salida: Posicion
    llegada: Posicion

    def dentro_de_limites(self, pos: Posicion) -> bool:
        """Verifica que una posición esté dentro de los límites de la matriz.

        El motor de ejecución debe invocar siempre esta función antes de
        indexar la matriz, para evitar errores de indexación al intentar
        salir del laberinto.
        """
        i, j = pos
        return 0 <= i < self.num_filas and 0 <= j < self.num_columnas

    def es_muro(self, pos: Posicion) -> bool:
        i, j = pos
        return self.cuadricula[i][j] == MURO

    def es_transitable(self, pos: Posicion) -> bool:
        """Una celda es transitable si está dentro de los límites y no es muro.

        Las celdas con símbolo 0, 1 o 2 se consideran transitables.
        """
        return self.dentro_de_limites(pos) and not self.es_muro(pos)


def _leer_cuadricula_cruda(ruta: str | Path) -> List[List[str]]:
    """Lee el archivo CSV y retorna una matriz de strings, validando forma rectangular."""
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
    """Garantiza L[i,j] = X si i=1 v i=m v j=1 v j=r (en notación 1-indexada);
    equivalente 0-indexado: i=0 v i=m-1 v j=0 v j=r-1."""
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
    """Valida i_s = 2 y i_z = m-1 en notación 1-indexada del enunciado.

    Equivalente 0-indexado: fila de salida == 1, fila de llegada == m-2.
    """
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
    """Pertenece al conjunto interior I = {2,...,m-1} x {2,...,r-1} (1-indexado),
    equivalente 0-indexado: {1,...,m-2} x {1,...,r-2}."""
    i, j = pos
    return 1 <= i <= m - 2 and 1 <= j <= r - 2


def _validar_zona_despejada(
    cuadricula: Tuple[Tuple[str, ...], ...], pos: Posicion, m: int, r: int, etiqueta: str
) -> None:
    """Valida que N_8(pos) ∩ I no contenga muros, para pos en {salida, llegada}."""
    for vecino in _vecinos_moore(pos):
        if _es_interior(vecino, m, r) and cuadricula[vecino[0]][vecino[1]] == MURO:
            raise ErrorValidacionLaberinto(
                f"La zona interior alrededor de {etiqueta} (posición 0-indexada {pos}) "
                f"contiene un muro en la celda vecina interior {vecino}. "
                "El vecindario de Moore interior debe estar despejado."
            )


def cargar_laberinto_csv(ruta: str | Path) -> Laberinto:
    """Carga, valida y retorna un laberinto a partir de un archivo CSV.

    Lanza FileNotFoundError o ErrorValidacionLaberinto con mensajes descriptivos
    ante cualquier incumplimiento de las reglas estrictas de la pauta.
    """
    crudo = _leer_cuadricula_cruda(ruta)
    cuadricula: Tuple[Tuple[str, ...], ...] = tuple(tuple(fila) for fila in crudo)
    m = len(cuadricula)
    r = len(cuadricula[0])

    if m < 5 or r < 5:
        # Se requiere al menos 1 fila de perímetro + 1 fila interior a cada lado
        # más separación entre salida y llegada; 5x5 es el mínimo estructuralmente
        # razonable (perímetro + salida + al menos una fila intermedia + llegada + perímetro).
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