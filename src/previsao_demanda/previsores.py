"""Previsores pontuais: duas linhas de base e dois modelos.

Convenção usada em todo o projeto: a **origem** t é o número de semanas já observadas. Na origem
t conhecemos y[0], …, y[t−1] e prevemos os horizontes h = 1, …, H, isto é, as semanas de índice
t + h − 1. Nenhuma função abaixo lê y[t] em diante para prever a partir da origem t (há um
teste que altera o futuro e confere que a previsão não muda).

Todas devolvem previsões na escala original, com forma (origens, horizontes, séries).
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from previsao_demanda import calendario

PERIODO = 52  # semanas por ano, para a sazonalidade
REAJUSTE = 13  # os modelos são reestimados a cada 13 semanas; entre um ajuste e outro só os dados andam
JANELA_NIVEL = 8  # semanas usadas para medir o nível recente da série
JANELA_EVENTO = 6  # a defasagem alinhada a um evento só vale até 6 semanas de distância dele
MENOR_ORIGEM_GBM = 60  # antes disso não há um ano de história para montar as defasagens


def _origem_do_ajuste(origem: int, reajuste: int) -> int:
    """Última origem de reestimação que não passa de `origem` (depende só da própria origem)."""
    return origem - origem % reajuste


# ---------------------------------------------------------------- linhas de base


def sazonal_ingenua(y: np.ndarray, origens: np.ndarray, horizonte_max: int, periodo: int = PERIODO) -> np.ndarray:
    """A mesma semana do ano anterior: a previsão para a semana k é y[k − 52]."""
    previsao = np.empty((len(origens), horizonte_max, y.shape[1]))
    for i, origem in enumerate(origens):
        for h in range(1, horizonte_max + 1):
            previsao[i, h - 1] = y[origem + h - 1 - periodo]
    return previsao


def media_movel(y: np.ndarray, origens: np.ndarray, horizonte_max: int, janela: int = 4) -> np.ndarray:
    """A média das últimas `janela` semanas, repetida para todos os horizontes."""
    previsao = np.empty((len(origens), horizonte_max, y.shape[1]))
    for i, origem in enumerate(origens):
        previsao[i] = y[origem - janela : origem].mean(axis=0)
    return previsao


# ---------------------------------------------------------------- suavização exponencial


def _modelo_ets(serie_log: np.ndarray, periodo: int) -> ETSModel:
    return ETSModel(
        serie_log, error="add", trend=None, seasonal="add", seasonal_periods=periodo, initialization_method="heuristic"
    )


def desvio_parametrico(variancia_do_residuo: float, suavizacao_do_nivel: float, horizontes: np.ndarray) -> np.ndarray:
    """Desvio-padrão do erro de previsão que o modelo ETS(A,N,A) declara para cada horizonte.

    Para horizontes menores que o período sazonal: var(h) = σ² · [1 + (h − 1) · α²], com σ² a
    variância do resíduo de um passo e α a suavização do nível. É daqui que sai o intervalo
    paramétrico (normal), o que o modelo "promete".
    """
    return np.sqrt(variancia_do_residuo * (1 + (np.asarray(horizontes) - 1) * suavizacao_do_nivel**2))


def suavizacao_exponencial(
    y: np.ndarray, origens: np.ndarray, horizonte_max: int, periodo: int = PERIODO, reajuste: int = REAJUSTE
) -> tuple[np.ndarray, np.ndarray]:
    """Suavização exponencial com sazonalidade aditiva sobre log(1 + y): ETS(A,N,A) do statsmodels.

    Os parâmetros de suavização são estimados na última origem de reajuste; nas origens
    seguintes o modelo só é atualizado com as semanas novas. Devolve a previsão na escala
    original e o desvio-padrão declarado pelo modelo, na escala log, para cada horizonte.
    """
    log = np.log1p(y)
    horizontes = np.arange(1, horizonte_max + 1)
    previsao = np.empty((len(origens), horizonte_max, y.shape[1]))
    desvio = np.empty_like(previsao)
    parametros: dict[tuple[int, int], np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, origem in enumerate(origens):
            ajuste_em = _origem_do_ajuste(int(origem), reajuste)
            for s in range(y.shape[1]):
                if (ajuste_em, s) not in parametros:
                    parametros[(ajuste_em, s)] = _modelo_ets(log[:ajuste_em, s], periodo).fit(disp=False).params
                resultado = _modelo_ets(log[:origem, s], periodo).smooth(parametros[(ajuste_em, s)])
                previsao[i, :, s] = np.expm1(resultado.forecast(horizonte_max))
                desvio[i, :, s] = desvio_parametrico(resultado.mse, resultado.smoothing_level, horizontes)
    return previsao, desvio


# ---------------------------------------------------------------- gradient boosting


def _media_da_janela(acumulado: np.ndarray, inicio: np.ndarray, fim: np.ndarray) -> np.ndarray:
    """Média de log[inicio:fim] por série, a partir da soma acumulada (com zero na frente)."""
    return (acumulado[fim] - acumulado[inicio]) / (fim - inicio)[:, None]


def montar_atributos(log: np.ndarray, semanas: pd.DatetimeIndex, origens: np.ndarray, h: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Atributos para prever a semana origem + h − 1, usando só o que se conhece na origem.

    Tudo é medido em relação ao nível recente da série (média do log das últimas 8 semanas),
    para que um artigo grande e um pequeno falem a mesma língua. Devolve a tabela de atributos
    (uma linha por origem e série) e o nível de cada linha, que é somado de volta na previsão.

    Grupos de atributos:
    - defasagens recentes e tendência curta;
    - a mesma semana do ano anterior (e as vizinhas) e o salto que ela deu sobre o nível da época;
    - calendário da semana-alvo: semana do ano e distância até cada data comemorativa;
    - defasagem alinhada ao evento: o valor da semana que, no ano anterior, estava à mesma
      distância da data (o que a "mesma semana do ano anterior" erra quando a data é móvel).
    """
    origens = np.asarray(origens)
    n_series = log.shape[1]
    alvo = origens + h - 1
    acumulado = np.vstack([np.zeros((1, n_series)), np.cumsum(log, axis=0)])
    nivel = _media_da_janela(acumulado, origens - JANELA_NIVEL, origens)

    def relativo(posicao: np.ndarray) -> np.ndarray:
        """log[posicao] − nível, com NaN onde a posição não existe ou ainda não foi observada."""
        valida = (posicao >= 0) & (posicao < origens)
        valores = log[np.where(valida, posicao, 0)] - nivel
        valores[~valida] = np.nan
        return valores

    colunas: dict[str, np.ndarray] = {}
    for defasagem in (1, 2, 3, 4):
        colunas[f"defasagem_{defasagem}"] = relativo(origens - defasagem)
    metade = JANELA_NIVEL // 2
    colunas["tendencia_curta"] = _media_da_janela(acumulado, origens - metade, origens) - _media_da_janela(
        acumulado, origens - JANELA_NIVEL, origens - metade
    )
    for defasagem in (PERIODO - 1, PERIODO, PERIODO + 1, 2 * PERIODO):
        colunas[f"sazonal_{defasagem}"] = relativo(alvo - defasagem)
    nivel_ano_passado = _media_da_janela(acumulado, origens - PERIODO - JANELA_NIVEL, origens - PERIODO)
    colunas["salto_ano_passado"] = log[alvo - PERIODO] - nivel_ano_passado
    colunas["nivel_ano_passado"] = nivel_ano_passado - nivel

    # O calendário é conhecido para sempre: pode ser calculado para a semana-alvo, mesmo no futuro.
    semanas_alvo = semanas[0] + pd.to_timedelta(7 * alvo, unit="D")
    calendario_alvo: dict[str, np.ndarray] = {"semana_do_ano": semanas_alvo.isocalendar().week.to_numpy().astype(float)}
    for evento in calendario.EVENTOS:
        distancia = calendario.semanas_ate_evento(semanas_alvo, evento)
        calendario_alvo[f"semanas_ate_{evento}"] = np.clip(distancia, -12, 12).astype(float)
        posicao = calendario.semana_alinhada_do_ano_anterior(semanas_alvo, evento, semanas[0])
        posicao = np.where(np.abs(distancia) <= JANELA_EVENTO, posicao, -1)
        colunas[f"alinhada_{evento}"] = relativo(posicao)

    tabela = {nome: valores.reshape(-1) for nome, valores in colunas.items()}
    for nome, valores in calendario_alvo.items():
        tabela[nome] = np.repeat(valores, n_series)
    tabela["artigo"] = np.tile(np.arange(n_series), len(origens)).astype(float)  # categórica, tratada no treino
    return pd.DataFrame(tabela), nivel


ARVORES_GBM = 100
PARAMETROS_GBM = {
    "objective": "l1",  # o erro é medido em valor absoluto; a perda L1 mira a mediana
    "learning_rate": 0.1,
    "num_leaves": 15,
    # Uma data comemorativa acontece uma vez por ano: são poucas linhas por artigo. Com folhas de no mínimo
    # 20 linhas (o valor inicial) o modelo não consegue isolar a semana do pico; um teste sintético mostrou
    # isso. O valor 5 foi escolhido nas origens de aquecimento, antes de reavaliar na janela de avaliação.
    "min_data_in_leaf": 5,
    "max_bin": 63,  # poucas linhas (alguns milhares): um histograma mais grosso basta e treina mais rápido
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "seed": 42,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def gradient_boosting(
    y: np.ndarray,
    semanas: pd.DatetimeIndex,
    origens: np.ndarray,
    horizonte_max: int,
    reajuste: int = REAJUSTE,
    parametros: dict | None = None,
    arvores: int = ARVORES_GBM,
) -> np.ndarray:
    """LightGBM global (um modelo para todos os artigos) e direto (um modelo por horizonte).

    O alvo é log(1 + y) da semana prevista menos o nível recente da série. O modelo de cada
    horizonte é treinado, na origem de reajuste T, só com pares (origem, alvo) cujo alvo já foi
    observado em T, e prevê as origens até o reajuste seguinte.

    Os atributos de todas as origens são montados uma vez por horizonte. Isso não vaza: os
    atributos da origem t só leem semanas anteriores a t, e o alvo de cada linha só entra no
    treino quando já foi observado na origem de reajuste.
    """
    parametros = {**PARAMETROS_GBM, **(parametros or {})}
    log = np.log1p(y)
    origens = np.asarray(origens)
    n_series = y.shape[1]
    previsao = np.empty((len(origens), horizonte_max, n_series))
    ajustes = np.array([_origem_do_ajuste(int(origem), reajuste) for origem in origens])
    todas = np.arange(MENOR_ORIGEM_GBM, origens.max() + 1)
    for h in range(1, horizonte_max + 1):
        tabela, nivel = montar_atributos(log, semanas, todas, h)
        atributos = tabela.to_numpy(dtype=float)
        categorica = [tabela.columns.get_loc("artigo")]
        for ajuste_em in np.unique(ajustes):
            n_treino = ajuste_em - h + 1 - MENOR_ORIGEM_GBM  # origens cujo alvo (origem + h − 1) é ≤ ajuste_em − 1
            if n_treino * n_series < 200:
                raise ValueError("História curta demais para treinar o gradient boosting nesta origem.")
            origens_treino = todas[:n_treino]
            alvo = (log[origens_treino + h - 1] - nivel[:n_treino]).reshape(-1)
            treino = lgb.Dataset(
                atributos[: n_treino * n_series], label=alvo, feature_name=list(tabela.columns), categorical_feature=categorica
            )
            modelo = lgb.train(parametros, treino, num_boost_round=arvores)
            daqui = np.flatnonzero(ajustes == ajuste_em)
            posicao = origens[daqui] - MENOR_ORIGEM_GBM
            linhas = (posicao[:, None] * n_series + np.arange(n_series)[None, :]).reshape(-1)
            predito = modelo.predict(atributos[linhas]).reshape(len(daqui), n_series)
            previsao[daqui, h - 1] = np.expm1(predito + nivel[posicao])
    return previsao
