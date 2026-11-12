import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from previsao_demanda import calendario, previsores, validacao

INICIO = pd.Timestamp("2015-07-06")


def _series_sazonais(n: int = 230, series: int = 3, semente: int = 0) -> np.ndarray:
    """Séries positivas com sazonalidade anual e ruído multiplicativo."""
    gerador = np.random.default_rng(semente)
    t = np.arange(n)[:, None]
    nivel = np.array([500.0, 2000.0, 8000.0])[:series]
    sazonal = 1 + 0.6 * np.sin(2 * np.pi * t / 52 + np.arange(series))
    return nivel * sazonal * np.exp(gerador.normal(0, 0.1, size=(n, series)))


def test_sazonal_ingenua_acerta_em_cheio_uma_serie_periodica():
    t = np.arange(260)
    y = (100 + 50 * np.sin(2 * np.pi * t / 52))[:, None]
    origens = validacao.origens_moveis(len(y), 104, 8)
    previsao = previsores.sazonal_ingenua(y, origens, 8)
    np.testing.assert_allclose(previsao, validacao.realizado(y, origens, 8), atol=1e-9)


def test_sazonal_ingenua_e_media_movel_em_valores_conhecidos():
    y = np.arange(200.0)[:, None]  # y[k] = k
    origens = np.array([100, 150])
    sazonal = previsores.sazonal_ingenua(y, origens, 3)
    # Origem 100, horizonte 1 → semana 100 → valor da semana 48.
    np.testing.assert_array_equal(sazonal[:, :, 0], [[48, 49, 50], [98, 99, 100]])
    media = previsores.media_movel(y, origens, 3, janela=4)
    # Média das semanas 96 a 99 = 97,5, repetida nos três horizontes.
    np.testing.assert_array_equal(media[:, :, 0], [[97.5] * 3, [147.5] * 3])


def test_nenhum_previsor_enxerga_o_futuro():
    """Trocar todos os valores a partir da origem não pode mudar a previsão feita naquela origem."""
    y = _series_sazonais()
    semanas = pd.date_range(INICIO, periods=len(y), freq="7D")
    origens = np.array([200, 205])
    futuro_trocado = y.copy()
    futuro_trocado[200:] = 1e6
    funcoes = {
        "sazonal": lambda dados: previsores.sazonal_ingenua(dados, origens, 8),
        "media_movel": lambda dados: previsores.media_movel(dados, origens, 8),
        "ets": lambda dados: previsores.suavizacao_exponencial(dados, origens, 8)[0],
        "ets_desvio": lambda dados: previsores.suavizacao_exponencial(dados, origens, 8)[1],
        "gbm": lambda dados: previsores.gradient_boosting(dados, semanas, origens, 8, arvores=20),
    }
    for nome, funcao in funcoes.items():
        original, trocado = funcao(y), funcao(futuro_trocado)
        np.testing.assert_array_equal(original[0], trocado[0], err_msg=f"{nome} usou dado do futuro na origem 200")
        # A origem 205 já viu cinco semanas trocadas: aí a previsão tem de mudar (o teste não é vazio).
        # A sazonal ingênua é a exceção: ela só olha 52 semanas atrás, que não foram trocadas.
        if nome != "sazonal":
            assert not np.array_equal(original[1], trocado[1]), nome
    # Para a sazonal ingênua, o teste não vazio é trocar o ano anterior.
    passado_trocado = y.copy()
    passado_trocado[140:160] = 1e6
    assert not np.array_equal(funcoes["sazonal"](y), funcoes["sazonal"](passado_trocado))


def test_atributos_da_origem_so_usam_semanas_anteriores():
    y = _series_sazonais()
    semanas = pd.date_range(INICIO, periods=len(y), freq="7D")
    log, trocado = np.log1p(y), np.log1p(y)
    trocado = trocado.copy()
    trocado[180:] = 99.0
    origens = np.array([120, 180])
    for h in (1, 8):
        tabela, nivel = previsores.montar_atributos(log, semanas, origens, h)
        tabela_trocada, nivel_trocado = previsores.montar_atributos(trocado, semanas, origens, h)
        pd.testing.assert_frame_equal(tabela, tabela_trocada)
        np.testing.assert_array_equal(nivel, nivel_trocado)
    assert len(tabela) == len(origens) * y.shape[1]
    assert {"sazonal_52", "semanas_ate_pascoa", "alinhada_pascoa", "alinhada_black_friday", "artigo"} <= set(tabela.columns)


def test_desvio_parametrico_bate_com_o_intervalo_do_statsmodels():
    """A fórmula σ·√(1 + (h−1)·α²) reproduz o intervalo de previsão analítico do ETS(A,N,A)."""
    y = np.log1p(_series_sazonais(n=220, series=1, semente=3)[:, 0])
    serie = pd.Series(y, index=pd.date_range(INICIO, periods=len(y), freq="W-MON"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste = ETSModel(
            serie, error="add", trend=None, seasonal="add", seasonal_periods=52, initialization_method="heuristic"
        ).fit(disp=False)
    faixa = ajuste.get_prediction(start=len(y), end=len(y) + 7).pred_int(alpha=0.20)
    meia_largura = ((faixa.iloc[:, 1] - faixa.iloc[:, 0]) / 2).to_numpy()
    desvio = previsores.desvio_parametrico(ajuste.mse, ajuste.smoothing_level, np.arange(1, 9))
    np.testing.assert_allclose(1.2815515655446004 * desvio, meia_largura, rtol=1e-6)
    assert (np.diff(desvio) >= 0).all()  # a incerteza declarada não diminui com o horizonte


def test_suavizacao_exponencial_segue_uma_sazonalidade_estavel():
    y = _series_sazonais(n=270, semente=5)
    origens = validacao.origens_moveis(len(y), 208, 8)
    previsao, desvio = previsores.suavizacao_exponencial(y, origens, 8)
    real = validacao.realizado(y, origens, 8)
    erro_relativo = np.median(np.abs(previsao - real) / real)
    assert erro_relativo < 0.15  # ruído de 10% por semana; a média móvel erraria a sazonalidade inteira
    erro_da_media = np.median(np.abs(previsores.media_movel(y, origens, 8) - real) / real)
    assert erro_relativo < erro_da_media
    assert (desvio > 0).all()


def test_gradient_boosting_aprende_uma_data_movel_que_a_sazonal_ingenua_erra():
    """Pico na semana da Páscoa, que muda de semana a cada ano: quem olha 52 semanas atrás erra o pico."""
    gerador = np.random.default_rng(7)
    n = 420
    semanas = pd.date_range(INICIO, periods=n, freq="7D")
    distancia = calendario.semanas_ate_evento(semanas, "pascoa")
    pico = np.where(distancia == 0, 20.0, np.where(distancia == 1, 5.0, 1.0))[:, None]
    y = np.array([800.0, 3000.0]) * pico * np.exp(gerador.normal(0, 0.08, size=(n, 2)))
    origens = validacao.origens_moveis(n, 340, 4)
    real = validacao.realizado(y, origens, 4)
    gbm = previsores.gradient_boosting(y, semanas, origens, 4)
    sazonal = previsores.sazonal_ingenua(y, origens, 4)
    na_pascoa = (distancia == 0)[origens[:, None] + np.arange(4)[None, :]]
    assert na_pascoa.sum() >= 8  # duas Páscoas na janela, vistas de quatro horizontes
    erro_gbm = np.abs(gbm - real)[na_pascoa].mean()
    erro_sazonal = np.abs(sazonal - real)[na_pascoa].mean()
    assert erro_gbm < 0.6 * erro_sazonal
    # Com folhas de no mínimo 20 linhas o modelo não isola a semana do pico (são 2 artigos e poucas Páscoas):
    # foi esse caso que levou a reduzir o tamanho mínimo da folha.
    folha_grande = previsores.gradient_boosting(y, semanas, origens, 4, parametros={"min_data_in_leaf": 20})
    assert erro_gbm < 0.7 * np.abs(folha_grande - real)[na_pascoa].mean()


def test_gradient_boosting_recusa_historia_curta():
    y = _series_sazonais(n=120)
    semanas = pd.date_range(INICIO, periods=len(y), freq="7D")
    with pytest.raises(ValueError):
        previsores.gradient_boosting(y, semanas, np.array([70]), 8, arvores=5)
