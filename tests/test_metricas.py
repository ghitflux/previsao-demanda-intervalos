import numpy as np
import pytest

from previsao_demanda import metricas


def test_escala_mase_em_serie_conhecida():
    # Série que sobe 2 a cada passo: com período 4, a sazonal ingênua erra sempre 8.
    serie = np.arange(0.0, 40.0, 2.0)
    assert metricas.escala_mase(serie, periodo=4) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        metricas.escala_mase(serie[:4], periodo=4)


def test_mase_vale_um_quando_o_erro_e_igual_a_escala_e_pesa_as_series_por_igual():
    escalas = np.array([10.0, 1000.0])
    erros = np.tile(escalas, (30, 1))  # cada série erra exatamente a sua escala
    assert metricas.mase(erros, escalas) == pytest.approx(1.0)
    # Dobrar o erro só da série pequena sobe o MASE para 1,5: a série grande não domina.
    erros[:, 0] *= 2
    assert metricas.mase(erros, escalas) == pytest.approx(1.5)
    assert metricas.mase_por_origem(erros, escalas).shape == (30,)


def test_wape_pondera_pelo_volume():
    reais = np.array([[100.0, 1000.0], [100.0, 1000.0]])
    erros = np.array([[50.0, 100.0], [50.0, 100.0]])  # 50% na série pequena, 10% na grande
    assert metricas.wape(erros, reais) == pytest.approx(300 / 2200)


def test_indices_em_blocos_sao_blocos_de_origens_consecutivas():
    indices = metricas.indices_em_blocos(n=100, bloco=13, repeticoes=50, semente=0)
    assert indices.shape == (50, 100)
    assert indices.min() >= 0 and indices.max() <= 99
    # Dentro de cada bloco de 13 os índices andam de um em um.
    passos = np.diff(indices[:, :91].reshape(50, 7, 13), axis=2)
    assert (passos == 1).all()
    # A mesma semente devolve a mesma reamostragem.
    np.testing.assert_array_equal(indices, metricas.indices_em_blocos(100, 13, 50, semente=0))
    with pytest.raises(ValueError):
        metricas.indices_em_blocos(10, 11, 5)


def test_intervalo_bootstrap_da_media_bate_com_a_teoria_em_dados_independentes():
    gerador = np.random.default_rng(0)
    valores = gerador.normal(10.0, 2.0, size=400)
    indices = metricas.indices_em_blocos(400, bloco=1, repeticoes=4000, semente=1)
    inferior, superior = metricas.intervalo_bootstrap(valores, indices)
    erro_padrao = valores.std(ddof=1) / np.sqrt(400)
    assert inferior == pytest.approx(valores.mean() - 1.96 * erro_padrao, abs=0.03)
    assert superior == pytest.approx(valores.mean() + 1.96 * erro_padrao, abs=0.03)


def test_bloco_maior_alarga_o_intervalo_quando_ha_autocorrelacao():
    gerador = np.random.default_rng(2)
    ruido = gerador.normal(size=600)
    serie = np.convolve(ruido, np.ones(8) / 8, mode="same")  # vizinhos parecidos, como erros de origens seguidas
    solto = metricas.intervalo_bootstrap(serie, metricas.indices_em_blocos(600, 1, 3000, semente=3))
    em_blocos = metricas.intervalo_bootstrap(serie, metricas.indices_em_blocos(600, 13, 3000, semente=3))
    assert (em_blocos[1] - em_blocos[0]) > 1.5 * (solto[1] - solto[0])


def test_razao_pareada_de_um_modelo_com_ele_mesmo_e_um():
    gerador = np.random.default_rng(4)
    erros = gerador.exponential(size=200)
    indices = metricas.indices_em_blocos(200, 13, 500, semente=5)
    assert metricas.intervalo_bootstrap(erros, indices, erros) == (1.0, 1.0)
    # Um modelo que erra sempre a metade tem razão 0,5 sem incerteza: o pareamento cancela o ruído comum.
    inferior, superior = metricas.intervalo_bootstrap(erros / 2, indices, erros)
    assert inferior == pytest.approx(0.5) and superior == pytest.approx(0.5)
