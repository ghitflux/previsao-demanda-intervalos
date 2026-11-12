"""Intervalos de previsão por predição conforme, em NumPy.

A ideia: em vez de confiar na variância que o modelo declara, olhar para o tamanho dos erros
que ele cometeu de verdade em previsões passadas (fora da amostra) e usar o quantil desses erros
como raio do intervalo. Não supõe distribuição nenhuma para o erro.

Três versões:
- **simples (split conformal):** o raio é o quantil dos erros passados. Garante a cobertura
  quando os erros são permutáveis (por exemplo, independentes e com a mesma distribuição).
- **adaptativa (ACI, Gibbs e Candès, 2021):** séries temporais mudam de regime e quebram a
  permutabilidade. A ACI corrige o nível pedido a cada passo: errou, pede um intervalo mais
  largo; acertou, aperta um pouco. A cobertura de longo prazo fica garantida para qualquer
  sequência, sem suposição sobre os dados.
- **por grupo (Mondrian):** a garantia das duas primeiras vale na média de todas as previsões.
  Calibrando cada grupo de previsões com os erros do próprio grupo, ela passa a valer dentro
  de cada grupo.

Em previsão com horizonte h, o erro da previsão feita na origem t só é conhecido h passos
depois. As duas funções recebem esse `atraso` e nunca usam um erro antes de ele existir.
"""

from __future__ import annotations

from math import ceil

import numpy as np


def quantil_conforme(pontuacoes: np.ndarray, alfa: float) -> float:
    """Raio do intervalo: o ⌈(n+1)(1−alfa)⌉-ésimo menor erro de calibração.

    O "+1" é a correção de amostra finita que dá a garantia de cobertura de pelo menos 1−alfa.
    Se o nível pedido passa do que n erros permitem afirmar (ou alfa ≤ 0), o raio é infinito;
    se alfa ≥ 1, o intervalo é vazio (raio −infinito).
    """
    n = len(pontuacoes)
    if alfa >= 1:
        return -np.inf
    if alfa <= 0 or n == 0:
        return np.inf
    posicao = ceil((n + 1) * (1 - alfa))
    if posicao > n:
        return np.inf
    return float(np.partition(pontuacoes, posicao - 1)[posicao - 1])


def _conhecidas(pontuacoes: np.ndarray, passo: int, atraso: int, janela: int | None) -> np.ndarray:
    """Erros já observados no passo `passo`: os das previsões feitas até `passo − atraso`."""
    fim = passo - atraso + 1
    if fim <= 0:
        return pontuacoes[:0]
    inicio = 0 if janela is None else max(0, fim - janela)
    return pontuacoes[inicio:fim]


def conformal_simples(
    pontuacoes: np.ndarray, alfa: float, atraso: int = 1, minimo: int = 30, janela: int | None = None
) -> np.ndarray:
    """Raio do intervalo em cada passo, pelo quantil dos erros conhecidos até ali.

    `pontuacoes[i]` é o erro absoluto da previsão feita no passo i, que só fica conhecido no
    passo i + atraso. Enquanto houver menos de `minimo` erros conhecidos, o raio é NaN (sem
    intervalo). `janela` limita a calibração aos erros mais recentes; None usa todos.
    """
    pontuacoes = np.asarray(pontuacoes, dtype=float)
    raios = np.full(len(pontuacoes), np.nan)
    for passo in range(len(pontuacoes)):
        conhecidas = _conhecidas(pontuacoes, passo, atraso, janela)
        if len(conhecidas) >= minimo:
            raios[passo] = quantil_conforme(conhecidas, alfa)
    return raios


def conformal_adaptativo(
    pontuacoes: np.ndarray,
    alfa: float,
    gama: float = 0.02,
    atraso: int = 1,
    minimo: int = 30,
    janela: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ACI: o mesmo quantil, mas com o nível corrigido a cada erro observado.

    alfa_t+1 = alfa_t + gama · (alfa − errou_t), em que errou_t é 1 se o intervalo do passo t não
    cobriu o valor real. Errar baixa alfa_t (intervalo mais largo); acertar sobe (mais estreito).
    Com horizonte h, a correção referente ao passo t só entra no passo t + atraso.
    Devolve os raios e a trajetória de alfa_t.
    """
    pontuacoes = np.asarray(pontuacoes, dtype=float)
    n = len(pontuacoes)
    raios = np.full(n, np.nan)
    niveis = np.full(n, np.nan)
    alfa_t = alfa
    for passo in range(n):
        anterior = passo - atraso
        if anterior >= 0 and np.isfinite(niveis[anterior]):
            errou = float(pontuacoes[anterior] > raios[anterior])
            alfa_t += gama * (alfa - errou)
        conhecidas = _conhecidas(pontuacoes, passo, atraso, janela)
        if len(conhecidas) >= minimo:
            niveis[passo] = alfa_t
            raios[passo] = quantil_conforme(conhecidas, alfa_t)
    return raios, niveis


def conformal_por_grupo(
    pontuacoes: np.ndarray, grupos: np.ndarray, alfa: float, atraso: int = 1, minimo: int = 30
) -> np.ndarray:
    """Conformal com calibração separada por grupo (Mondrian): cada grupo usa só os erros do próprio grupo.

    A garantia do conformal simples vale na média de tudo. Se um grupo de previsões é muito mais
    difícil que o resto (as semanas de data comemorativa), a média esconde a falha: sobra
    cobertura nas semanas fáceis e falta nas difíceis. Calibrando por grupo, a garantia passa a
    valer dentro de cada um.

    `pontuacoes` e `grupos` têm forma (passos, séries). Previsões de séries diferentes com o
    mesmo rótulo de grupo dividem a mesma calibração; para calibrar por série, use um rótulo por
    série. O raio é NaN enquanto o grupo tiver menos de `minimo` erros conhecidos.
    """
    pontuacoes = np.asarray(pontuacoes, dtype=float)
    grupos = np.asarray(grupos)
    raios = np.full(pontuacoes.shape, np.nan)
    for passo in range(len(pontuacoes)):
        fim = passo - atraso + 1
        if fim <= 0:
            continue
        conhecidas, rotulos = pontuacoes[:fim], grupos[:fim]
        for grupo in np.unique(grupos[passo]):
            calibracao = conhecidas[rotulos == grupo]
            if len(calibracao) >= minimo:
                raios[passo, grupos[passo] == grupo] = quantil_conforme(calibracao, alfa)
    return raios


def cobertura(pontuacoes: np.ndarray, raios: np.ndarray) -> float:
    """Fração dos passos com intervalo em que o valor real ficou dentro dele."""
    validos = ~np.isnan(raios)
    return float(np.mean(pontuacoes[validos] <= raios[validos]))


def limite_da_aci(passos: int, alfa: float, gama: float) -> float:
    """Desvio máximo entre a taxa de erro da ACI e alfa após `passos` passos (sem atraso).

    Resultado de Gibbs e Candès (2021): vale para qualquer sequência de dados, porque alfa_t
    nunca sai de [−gama, 1 + gama].
    """
    return (max(alfa, 1 - alfa) + gama) / (gama * passos)
