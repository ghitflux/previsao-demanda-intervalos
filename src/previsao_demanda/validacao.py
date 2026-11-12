"""Validação por origem móvel (rolling-origin backtesting).

Em cada origem o modelo vê só o passado, prevê as próximas H semanas e é cobrado pelo que
aconteceu. A origem anda uma semana e o processo se repete, como aconteceria em produção.
"""

from __future__ import annotations

import numpy as np


def origens_moveis(n_semanas: int, primeira: int, horizonte_max: int) -> np.ndarray:
    """Origens de `primeira` até a última que ainda tem as H semanas seguintes observadas.

    A origem t significa "t semanas já observadas"; a mais distante prevista é a de índice
    t + H − 1, que precisa existir. Todas as origens são avaliadas em todos os horizontes.
    """
    ultima = n_semanas - horizonte_max
    if primeira > ultima:
        raise ValueError("Série curta demais para esse começo e esse horizonte.")
    return np.arange(primeira, ultima + 1)


def realizado(y: np.ndarray, origens: np.ndarray, horizonte_max: int) -> np.ndarray:
    """O que aconteceu em cada (origem, horizonte, série): y[origem + h − 1]."""
    posicoes = np.asarray(origens)[:, None] + np.arange(horizonte_max)[None, :]
    return y[posicoes]
