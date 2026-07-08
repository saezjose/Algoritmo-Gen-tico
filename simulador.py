from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from cargador_laberinto import Laberinto, Posicion

Cromosoma = Tuple[str, ...]

ACCIONES: Tuple[str, ...] = ("H", "A", "M", "Q")
DIRECCIONES: Tuple[str, ...] = ("N", "E", "S", "O")
VECTORES_DIRECCION = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "O": (0, -1),
}
DIRECCION_INICIAL = "S"


@dataclass(frozen=True)
class RegistroPaso:
    indice_paso: int  
    gen: str
    posicion_antes: Posicion  
    posicion_despues: Posicion 
    direccion_antes: str  
    direccion_despues: str  
    choco: bool 

@dataclass(frozen=True)
class ResultadoSimulacion:
    trayectoria: Tuple[RegistroPaso, ...]
    posicion_final: Posicion
    direccion_final: str


def _girar_horario(direccion: str) -> str:
    idx = DIRECCIONES.index(direccion)
    return DIRECCIONES[(idx + 1) % 4]


def _girar_antihorario(direccion: str) -> str:
    idx = DIRECCIONES.index(direccion)
    return DIRECCIONES[(idx - 1) % 4]


def simular_cromosoma(cromosoma: Cromosoma, laberinto: Laberinto) -> ResultadoSimulacion:
    posicion: Posicion = laberinto.salida
    direccion = DIRECCION_INICIAL
    trayectoria: List[RegistroPaso] = []

    for t, gen in enumerate(cromosoma, start=1):
        posicion_antes = posicion
        direccion_antes = direccion
        choco = False

        if gen == "H":
            direccion = _girar_horario(direccion)
        elif gen == "A":
            direccion = _girar_antihorario(direccion)
        elif gen == "Q":
            pass
        elif gen == "M":
            dr, dc = VECTORES_DIRECCION[direccion]
            posicion_tentativa: Posicion = (posicion[0] + dr, posicion[1] + dc)
            if laberinto.es_transitable(posicion_tentativa):
                posicion = posicion_tentativa
            else:
                choco = True
        else:
            raise ValueError(
                f"Gen inválido detectado en el cromosoma: {gen!r}. "
                f"Se esperaba uno de {ACCIONES}."
            )

        trayectoria.append(
            RegistroPaso(
                indice_paso=t,
                gen=gen,
                posicion_antes=posicion_antes,
                posicion_despues=posicion,
                direccion_antes=direccion_antes,
                direccion_despues=direccion,
                choco=choco,
            )
        )

    return ResultadoSimulacion(
        trayectoria=tuple(trayectoria),
        posicion_final=posicion,
        direccion_final=direccion,
    )