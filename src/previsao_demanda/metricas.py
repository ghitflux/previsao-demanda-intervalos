"""Métricas de erro de previsão e incerteza por bootstrap em blocos sobre as origens."""

from __future__ import annotations

import numpy as np


def escala_mase(serie: np.ndarray, periodo: int = 52) -> float:
    """Erro absoluto médio da previsão sazonal ingênua dentro da própria série (o denominador do MASE)."""
    serie = np.asarray(serie, dtype=float)
    if len(serie) <= periodo:
        raise ValueError("A série precisa ser mais longa que o período sazonal.")
    return float(np.mean(np.abs(serie[periodo:] - serie[:-periodo])))


def mase_por_origem(erros_absolutos: np.ndarray, escalas: np.ndarray) -> np.ndarray:
    """Contribuição de cada origem para o MASE: média, entre as séries, do erro dividido pela escala.

    `erros_absolutos` tem forma (origens, séries). A média desse vetor é o MASE.
    """
    return np.mean(erros_absolutos / escalas, axis=1)


def mase(erros_absolutos: np.ndarray, escalas: np.ndarray) -> float:
    """MASE médio entre as séries: erro absoluto médio de cada uma dividido pela sua escala.

    Abaixo de 1, a previsão erra menos do que a sazonal ingênua errava no período de referência.
    Cada série pesa igual, seja grande ou pequena.
    """
    return float(np.mean(mase_por_origem(erros_absolutos, escalas)))


def wape(erros_absolutos: np.ndarray, reais: np.ndarray) -> float:
    """Erro absoluto percentual ponderado: soma dos erros sobre a soma do realizado.

    Pondera pelo volume: semanas e séries com mais visualizações pesam mais.
    """
    return float(np.sum(erros_absolutos) / np.sum(np.abs(reais)))


def indices_em_blocos(n: int, bloco: int, repeticoes: int, semente: int = 42) -> np.ndarray:
    """Índices de reamostragem por blocos móveis: (repeticoes, n).

    Erros de origens vizinhas são correlacionados (a mesma semana de pico aparece em oito
    origens seguidas). Sortear origens soltas daria um intervalo estreito demais; aqui o sorteio
    é de blocos de origens consecutivas, que preservam essa dependência.
    """
    if not 1 <= bloco <= n:
        raise ValueError("O bloco precisa ter entre 1 e n origens.")
    gerador = np.random.default_rng(semente)
    quantos = int(np.ceil(n / bloco))
    inicios = gerador.integers(0, n - bloco + 1, size=(repeticoes, quantos))
    indices = inicios[:, :, None] + np.arange(bloco)[None, None, :]
    return indices.reshape(repeticoes, -1)[:, :n]


def intervalo_bootstrap(
    numerador: np.ndarray, indices: np.ndarray, denominador: np.ndarray | None = None, confianca: float = 0.95
) -> tuple[float, float]:
    """Intervalo percentil de uma razão de somas sobre as origens reamostradas.

    Todas as métricas do estudo têm essa forma: MASE e cobertura são médias de um valor por
    origem (denominador = 1); WAPE é soma dos erros sobre soma do realizado; a razão entre dois
    modelos é a soma do erro de um sobre a soma do erro do outro, com as mesmas origens
    sorteadas para os dois (bootstrap pareado).
    """
    numerador = np.asarray(numerador, dtype=float)
    denominador = np.ones_like(numerador) if denominador is None else np.asarray(denominador, dtype=float)
    valores = numerador[indices].sum(axis=1) / denominador[indices].sum(axis=1)
    cauda = (1 - confianca) / 2
    return float(np.quantile(valores, cauda)), float(np.quantile(valores, 1 - cauda))
