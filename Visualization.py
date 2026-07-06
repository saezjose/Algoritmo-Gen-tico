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

from evaluador import ResultadoEvaluacion
from algoritmo_genetico import Cromosoma, ResultadoAG
from cargador_laberinto import Laberinto


def graficar_mejor_objetivo(resultado_ag: ResultadoAG, directorio_salida: str, mostrar: bool = True) -> str:
    """Gráfica 1: mejor J(x) global por generación, en escala logarítmica."""
    fig, ax = plt.subplots(figsize=(8, 5))
    generaciones = range(1, len(resultado_ag.historial_mejor_J) + 1)
    ax.semilogy(
        generaciones, resultado_ag.historial_mejor_J,
        marker="o", markersize=3, linewidth=1.2, color="#1f77b4",
    )
    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor J(x) global (escala logarítmica)")
    ax.set_title("Evolución del mejor valor de la función objetivo")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()

    ruta = os.path.join(directorio_salida, "grafica_1_mejor_objetivo.png")
    fig.savefig(ruta, dpi=150)
    if mostrar:
        plt.show()
    else:
        plt.close(fig)
    return ruta


def graficar_proporcion_validas(resultado_ag: ResultadoAG, directorio_salida: str, mostrar: bool = True) -> str:
    """Gráfica 2: proporción de soluciones válidas en la población por generación."""
    fig, ax = plt.subplots(figsize=(8, 5))
    generaciones = range(1, len(resultado_ag.historial_proporcion_valida) + 1)
    ax.plot(
        generaciones, resultado_ag.historial_proporcion_valida,
        marker="o", markersize=3, linewidth=1.2, color="#2ca02c",
    )
    ax.set_xlabel("Generación")
    ax.set_ylabel("Proporción de soluciones válidas")
    ax.set_title("Proporción de soluciones válidas en la población por generación")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    ruta = os.path.join(directorio_salida, "grafica_2_proporcion_validas.png")
    fig.savefig(ruta, dpi=150)
    if mostrar:
        plt.show()
    else:
        plt.close(fig)
    return ruta


def _describir_pasos(evaluacion: ResultadoEvaluacion) -> str:
    if evaluacion.es_valido:
        return f"{evaluacion.tau} pasos hasta la llegada válida (cromosoma de {evaluacion.n} genes)"
    return f"no alcanza la meta de forma válida (cromosoma completo de {evaluacion.n} genes)"


def imprimir_reporte_mejores_cromosomas(resultado_ag: ResultadoAG) -> None:
    """Consola/Texto: lista detallada de cromosomas únicos con el mejor J(x)."""
    mejores_unicos = resultado_ag.cromosomas_unicos_mejores()

    print("=" * 78)
    print("CONSOLA / TEXTO — Cromosomas únicos con el mejor valor objetivo")
    print("=" * 78)
    print(f"Mejor valor de función objetivo encontrado: J* = {resultado_ag.mejor_evaluacion.J}")
    print(f"Fitness asociado:                          phi* = {resultado_ag.mejor_evaluacion.phi}")
    print(f"Cantidad de cromosomas únicos que alcanzan J*:  {len(mejores_unicos)}")
    print(f"Solución válida:                            {resultado_ag.mejor_evaluacion.es_valido}")
    print("-" * 78)
    for idx, (cromosoma, evaluacion) in enumerate(mejores_unicos.items(), start=1):
        print(f"[{idx}] Cromosoma: {''.join(cromosoma)}")
        print(f"      {_describir_pasos(evaluacion)}")
        print(
            f"      D(x)={evaluacion.D}  tau(x)={evaluacion.tau}  "
            f"choques={evaluacion.conteo_choques}  "
            f"pausas_intermedias={evaluacion.conteo_Q_intermedio}  "
            f"acciones_post_meta={evaluacion.conteo_acciones_post_meta}  "
            f"bloques_giro={list(evaluacion.bloques_giro)}"
        )
    print("=" * 78)


def imprimir_trayectoria_auditada(cromosoma: Cromosoma, evaluacion: ResultadoEvaluacion, laberinto: Laberinto) -> None:
    """Consola/Auditoría: trayectoria paso a paso en coordenadas (X, Y), 1-indexadas.

    X corresponde a la columna del mapa; Y corresponde a la fila.
    """
    print("-" * 78)
    print(f"CONSOLA / AUDITORÍA — trayectoria de {''.join(cromosoma)}")
    print("-" * 78)

    fila_inicio, columna_inicio = laberinto.salida
    print(f"Paso 0 (inicio): (X={columna_inicio + 1}, Y={fila_inicio + 1})  dirección=S")

    for registro in evaluacion.trayectoria:
        i, j = registro.posicion_despues
        nota = "  [CHOQUE: movimiento bloqueado]" if (registro.gen == "M" and registro.choco) else ""
        print(
            f"Paso {registro.indice_paso:>3}: gen={registro.gen}  "
            f"(X={j + 1}, Y={i + 1})  dirección={registro.direccion_despues}{nota}"
        )
    print("-" * 78)


def generar_reporte_completo(resultado_ag: ResultadoAG, directorio_salida: Optional[str] = "resultados",
                              mostrar_graficas: bool = True) -> None:
    """Genera el conjunto completo de resultados mínimos exigidos por la pauta."""
    if directorio_salida:
        os.makedirs(directorio_salida, exist_ok=True)
        ruta1 = graficar_mejor_objetivo(resultado_ag, directorio_salida, mostrar=mostrar_graficas)
        ruta2 = graficar_proporcion_validas(resultado_ag, directorio_salida, mostrar=mostrar_graficas)
        print(f"Gráfica 1 (mejor objetivo, escala log) guardada en: {ruta1}")
        print(f"Gráfica 2 (proporción de válidas) guardada en:      {ruta2}")
        print()

    imprimir_reporte_mejores_cromosomas(resultado_ag)
    print()

    mejores_unicos = resultado_ag.cromosomas_unicos_mejores()
    for cromosoma, evaluacion in mejores_unicos.items():
        imprimir_trayectoria_auditada(cromosoma, evaluacion, resultado_ag.laberinto)
        print()