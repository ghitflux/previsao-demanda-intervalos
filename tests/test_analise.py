import numpy as np
import pandas as pd
import pytest

from previsao_demanda import analise, dados, metricas, resumo_site, validacao

ARTIGOS = list(dados.ARTIGOS)  # os dez artigos reais (o calendário de eventos depende deles); os valores são sintéticos


def _backtest_sintetico(desvio_real: float = 0.3, desvio_declarado: float = 0.3, semente: int = 0) -> analise.Backtest:
    """Backtest de mentira com resposta conhecida: cada previsão erra por um fator exp(N(0, desvio_real)).

    Não roda modelo nenhum; serve para conferir as contas de cobertura e de erro.
    """
    gerador = np.random.default_rng(semente)
    n = 480
    semanas = pd.date_range("2015-07-06", periods=n, freq="7D")
    y = 1e5 * np.exp(gerador.normal(0, 0.2, size=(n, len(ARTIGOS))))
    origens = validacao.origens_moveis(n, analise.PRIMEIRA_ORIGEM, analise.HORIZONTE_MAX)
    real = validacao.realizado(y, origens, analise.HORIZONTE_MAX)
    previsoes = {modelo: real * np.exp(gerador.normal(0, desvio_real, size=real.shape)) for modelo in analise.MODELOS}
    previsoes["sazonal"] = real * np.exp(gerador.normal(0, 2 * desvio_real, size=real.shape))  # a linha de base erra o dobro
    avaliadas = origens >= analise.PRIMEIRA_ORIGEM + analise.AQUECIMENTO
    return analise.Backtest(
        semanas=semanas,
        artigos=ARTIGOS,
        y=y,
        origens=origens,
        real=real,
        previsoes=previsoes,
        desvio_ets=np.full(real.shape, desvio_declarado),
        escalas=np.array([metricas.escala_mase(y[:260, s]) for s in range(len(ARTIGOS))]),
        avaliadas=avaliadas,
        indices=metricas.indices_em_blocos(int(avaliadas.sum()), analise.BLOCO, 300, analise.SEMENTE),
    )


def test_cobertura_bate_com_o_prometido_quando_a_distribuicao_e_conhecida():
    """Erro log-normal com o desvio certo: paramétrico e conformes cobrem 80% e 95%."""
    bt = _backtest_sintetico()
    pont = analise.pontuacoes(bt, "ets")
    cobertura = analise.cobertura_dos_intervalos(bt, pont, analise.raios_dos_intervalos(bt, "ets"))
    for metodo in analise.METODOS:
        for nivel in analise.NIVEIS:
            linha = cobertura.loc[(metodo, nivel, 0)]
            assert linha["cobertura"] == pytest.approx(nivel, abs=0.03), (metodo, nivel)
            assert linha["cobertura_inf"] <= linha["cobertura"] <= linha["cobertura_sup"]
    # Largura: o fator do paramétrico de 80% é exp(1,2816 · 0,3).
    assert cobertura.loc[("parametrico", 0.80, 0), "fator_mediano"] == pytest.approx(np.exp(1.2816 * 0.3), rel=1e-3)
    assert cobertura.loc[("conformal", 0.80, 0), "fator_mediano"] == pytest.approx(np.exp(1.2816 * 0.3), rel=0.08)


def test_parametrico_com_desvio_subestimado_cobre_menos_e_o_conformal_nao():
    """O modelo declara metade do desvio real: a cobertura de 80% cai para P(|Z| ≤ 0,64) = 47,8%."""
    bt = _backtest_sintetico(desvio_real=0.3, desvio_declarado=0.15, semente=1)
    pont = analise.pontuacoes(bt, "ets")
    cobertura = analise.cobertura_dos_intervalos(bt, pont, analise.raios_dos_intervalos(bt, "ets"))
    assert cobertura.loc[("parametrico", 0.80, 0), "cobertura"] == pytest.approx(0.478, abs=0.02)
    assert cobertura.loc[("conformal", 0.80, 0), "cobertura"] == pytest.approx(0.80, abs=0.03)
    assert cobertura.loc[("aci", 0.80, 0), "cobertura"] == pytest.approx(0.80, abs=0.03)


def test_semanas_de_evento_marcam_a_semana_da_data_e_a_anterior():
    bt = _backtest_sintetico()
    marcada = analise.semanas_de_evento(bt)
    assert marcada.shape == bt.real.shape
    natal, pascoa, imposto = (ARTIGOS.index(a) for a in ("Natal", "Páscoa", "Imposto_de_renda"))
    assert np.isnan(marcada[:, :, imposto]).all()  # sem data calculável: fora do recorte
    # Natal de 2022 (domingo 25/12) está na semana de 19/12; a anterior é a de 12/12.
    for dia, esperado in (("2022-12-12", 1), ("2022-12-19", 1), ("2022-12-05", 0), ("2022-12-26", 0)):
        alvo = bt.semanas.get_loc(pd.Timestamp(dia))
        i = int(np.flatnonzero(bt.origens == alvo)[0])  # horizonte 1 dessa origem é a própria semana
        assert marcada[i, 0, natal] == esperado, dia
        assert marcada[i, 0, pascoa] == 0
    # Duas semanas por ano para cada artigo com uma data só.
    alvo_h1 = bt.semanas[bt.origens]
    assert marcada[alvo_h1.year == 2022, 0, natal].sum() == 2
    assert marcada[alvo_h1.year == 2022, 0, pascoa].sum() == 2


def test_comparacao_reconhece_o_modelo_que_erra_a_metade_e_o_empate():
    bt = _backtest_sintetico(semente=2)
    comparacao = analise.comparacao_com_a_linha_de_base(bt)
    assert comparacao.loc["sazonal", "razao_para_a_sazonal"] == 1.0
    assert comparacao.loc["sazonal", "veredito"] == "linha de base"
    # Erro log-normal com o dobro do desvio: o erro absoluto médio fica um pouco acima do dobro.
    assert comparacao.loc["gbm", "razao_para_a_sazonal"] == pytest.approx(0.47, abs=0.05)
    assert comparacao.loc["gbm", "veredito"] == "erra menos"
    # Dois modelos com a mesma distribuição de erro empatam.
    bt.previsoes["sazonal"] = bt.real * np.exp(np.random.default_rng(9).normal(0, 0.3, size=bt.real.shape))
    assert analise.comparacao_com_a_linha_de_base(bt).loc["ets", "veredito"] == "empate"
    tabela = analise.erro_por_horizonte(bt)
    assert len(tabela) == len(analise.MODELOS) * analise.HORIZONTE_MAX
    assert (tabela["mase_inf"] <= tabela["mase"]).all() and (tabela["mase"] <= tabela["mase_sup"]).all()


def test_bloco_para_o_site_segue_o_contrato():
    bt = _backtest_sintetico(semente=3)
    pont = analise.pontuacoes(bt, "ets")
    raios = analise.raios_dos_intervalos(bt, "ets")
    r = analise.Resultados(
        backtest=bt,
        base=analise.resumo_da_base(bt),
        comparacao=analise.comparacao_com_a_linha_de_base(bt),
        erro_por_horizonte=analise.erro_por_horizonte(bt),
        razao_por_horizonte={m: analise.razao_por_horizonte(bt, m) for m in ("gbm", "ets")},
        erro_por_artigo=analise.erro_por_artigo(bt),
        raios=raios,
        cobertura=analise.cobertura_dos_intervalos(bt, pont, raios),
        cobertura_por_ano=analise.cobertura_por_ano(bt, pont, raios),
        cobertura_em_eventos=analise.cobertura_em_eventos(bt, pont, raios),
        conformal_por_modelo=pd.DataFrame(),
        sensibilidade_ao_gama=pd.DataFrame(),
        sensibilidade_a_folha={},
        exemplo=analise.serie_de_exemplo(bt, raios, "Natal", 4),
        achados={},
    )
    r.achados = analise.achados(r)
    bloco = resumo_site.para_o_site(r)
    idiomas = {"pt", "en", "es"}
    for chave in ("numero_chave", "pergunta", "resultado", "detalhe", "dados"):
        assert set(bloco[chave]) == idiomas
    assert all(len(texto) <= 12 for texto in bloco["numero_chave"].values())
    grafico = bloco["grafico"]
    assert grafico["tipo"] in {"linhas", "barras"} and grafico["formato_y"] == "percentual"
    assert 1 <= len(grafico["series"]) <= 4
    for serie in grafico["series"]:
        assert serie["papel"] in {"principal", "base", "alternativa", "neutro"}
        assert set(serie["nome"]) == idiomas and len(serie["pontos"]) <= 80
        assert all(0 <= valor <= 1 for _, valor in serie["pontos"])
        assert all(len(serie["rotulos"][idioma]) == len(serie["pontos"]) for idioma in idiomas)
    # Vírgula decimal em português e espanhol, ponto em inglês.
    assert "," in bloco["dados"]["pt"] and "," in bloco["dados"]["es"] and "." in bloco["dados"]["en"].rstrip(".")
