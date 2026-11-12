import numpy as np
import pytest
from scipy import stats

from previsao_demanda import conformal


def test_quantil_conforme_em_caso_de_resposta_conhecida():
    pontuacoes = np.arange(1.0, 20.0)  # 19 erros: 1, 2, …, 19
    # ⌈(19 + 1) · 0,8⌉ = 16: o raio é o 16º menor erro.
    assert conformal.quantil_conforme(pontuacoes, 0.20) == 16.0
    # ⌈20 · 0,95⌉ = 19: o maior erro.
    assert conformal.quantil_conforme(pontuacoes, 0.05) == 19.0
    # Com 18 erros não dá para prometer 95%: ⌈19 · 0,95⌉ = 19 > 18.
    assert conformal.quantil_conforme(pontuacoes[:18], 0.05) == np.inf
    assert conformal.quantil_conforme(pontuacoes, 0.0) == np.inf
    assert conformal.quantil_conforme(pontuacoes, 1.0) == -np.inf
    assert conformal.quantil_conforme(np.array([]), 0.2) == np.inf


def test_quantil_conforme_nao_depende_da_ordem():
    gerador = np.random.default_rng(0)
    pontuacoes = gerador.exponential(size=57)
    assert conformal.quantil_conforme(pontuacoes, 0.1) == conformal.quantil_conforme(np.sort(pontuacoes)[::-1], 0.1)


@pytest.mark.parametrize("alfa", [0.20, 0.05])
def test_cobertura_exata_com_erros_permutaveis(alfa):
    """Com n erros i.i.d., a cobertura esperada é exatamente ⌈(n+1)(1−alfa)⌉ / (n+1). Com n = 19: 80% e 95%."""
    gerador = np.random.default_rng(1)
    n, repeticoes = 19, 20000
    calibracao = np.abs(gerador.standard_t(df=3, size=(repeticoes, n)))  # cauda pesada: não importa a distribuição
    novo = np.abs(gerador.standard_t(df=3, size=repeticoes))
    raios = np.array([conformal.quantil_conforme(linha, alfa) for linha in calibracao])
    cobertura = np.mean(novo <= raios)
    esperado = np.ceil((n + 1) * (1 - alfa)) / (n + 1)
    assert cobertura == pytest.approx(esperado, abs=0.01)


@pytest.mark.parametrize("nivel", [0.80, 0.95])
def test_conformal_simples_recupera_o_quantil_de_uma_distribuicao_conhecida(nivel):
    """Erros |N(0, 1)|: o raio converge para o quantil teórico e a cobertura para o nível pedido."""
    gerador = np.random.default_rng(2)
    pontuacoes = np.abs(gerador.normal(size=6000))
    raios = conformal.conformal_simples(pontuacoes, 1 - nivel, atraso=1, minimo=50)
    assert np.isnan(raios[:50]).all() and not np.isnan(raios[50:]).any()
    teorico = stats.norm.ppf(1 - (1 - nivel) / 2)
    assert raios[-1] == pytest.approx(teorico, abs=0.06)
    assert conformal.cobertura(pontuacoes, raios) == pytest.approx(nivel, abs=0.015)


def test_o_atraso_impede_usar_erro_que_ainda_nao_existe():
    """Com horizonte 4, o raio do passo i só pode depender dos erros até o passo i − 4."""
    gerador = np.random.default_rng(3)
    pontuacoes = np.abs(gerador.normal(size=300))
    alterado = pontuacoes.copy()
    alterado[200:] *= 50  # muda o "futuro" a partir do passo 200
    for funcao in (
        lambda p: conformal.conformal_simples(p, 0.2, atraso=4, minimo=20),
        lambda p: conformal.conformal_adaptativo(p, 0.2, gama=0.05, atraso=4, minimo=20)[0],
    ):
        original, novo = funcao(pontuacoes), funcao(alterado)
        # Até o passo 203 nenhum erro alterado ficou conhecido (o do passo 200 só chega no 204).
        np.testing.assert_array_equal(original[:204], novo[:204])
        assert not np.array_equal(original[204:], novo[204:])


def test_janela_limita_a_calibracao_aos_erros_recentes():
    pontuacoes = np.concatenate([np.full(100, 10.0), np.full(100, 1.0)])
    sem_janela = conformal.conformal_simples(pontuacoes, 0.2, minimo=20)
    com_janela = conformal.conformal_simples(pontuacoes, 0.2, minimo=20, janela=30)
    assert sem_janela[-1] == 10.0  # os erros antigos ainda dominam o quantil
    assert com_janela[-1] == 1.0  # só os 30 mais recentes entram


def test_aci_respeita_o_limite_teorico_em_qualquer_sequencia():
    """Sequência adversária: cada erro é maior que todos os anteriores, o conformal simples nunca cobre.

    A ACI garante |taxa de erro − alfa| ≤ (max(alfa, 1 − alfa) + gama) / (gama · T) sem supor nada.
    """
    pontuacoes = 1.01 ** np.arange(2000)
    alfa, gama, minimo = 0.2, 0.05, 20
    simples = conformal.conformal_simples(pontuacoes, alfa, minimo=minimo)
    assert conformal.cobertura(pontuacoes, simples) == 0.0
    raios, niveis = conformal.conformal_adaptativo(pontuacoes, alfa, gama=gama, minimo=minimo)
    passos = int(np.sum(~np.isnan(raios)))
    taxa_de_erro = 1 - conformal.cobertura(pontuacoes, raios)
    assert abs(taxa_de_erro - alfa) <= conformal.limite_da_aci(passos, alfa, gama)
    assert np.nanmin(niveis) >= -gama - 1e-12 and np.nanmax(niveis) <= 1 + gama + 1e-12


def test_aci_recupera_a_cobertura_depois_de_uma_mudanca_de_regime():
    """Os erros triplicam de tamanho no meio da série: o conformal simples fica para trás, a ACI se ajusta."""
    gerador = np.random.default_rng(4)
    pontuacoes = np.abs(gerador.normal(size=3000))
    pontuacoes[1500:] *= 3
    alfa = 0.2
    simples = conformal.conformal_simples(pontuacoes, alfa, minimo=50)
    adaptativo, _ = conformal.conformal_adaptativo(pontuacoes, alfa, gama=0.02, minimo=50)
    depois = slice(1500, 3000)
    assert conformal.cobertura(pontuacoes[depois], simples[depois]) < 0.62
    assert conformal.cobertura(pontuacoes[depois], adaptativo[depois]) == pytest.approx(0.80, abs=0.02)
    # Antes da mudança os dois cobrem o prometido.
    antes = slice(50, 1500)
    assert conformal.cobertura(pontuacoes[antes], simples[antes]) == pytest.approx(0.80, abs=0.03)
    assert conformal.cobertura(pontuacoes[antes], adaptativo[antes]) == pytest.approx(0.80, abs=0.03)


def test_aci_com_gama_zero_e_o_conformal_simples():
    gerador = np.random.default_rng(5)
    pontuacoes = np.abs(gerador.normal(size=400))
    simples = conformal.conformal_simples(pontuacoes, 0.1, atraso=3, minimo=25)
    adaptativo, niveis = conformal.conformal_adaptativo(pontuacoes, 0.1, gama=0.0, atraso=3, minimo=25)
    np.testing.assert_array_equal(simples, adaptativo)
    assert np.nanmax(np.abs(niveis - 0.1)) == 0


def test_conformal_por_grupo_cobre_dentro_de_cada_grupo():
    """Um grupo raro com erros cinco vezes maiores: a calibração única falha nele, a por grupo não."""
    gerador = np.random.default_rng(6)
    passos, series = 4000, 3
    dificil = gerador.random((passos, series)) < 0.05
    pontuacoes = np.abs(gerador.normal(size=(passos, series))) * np.where(dificil, 5.0, 1.0)
    alfa = 0.2
    unico = conformal.conformal_por_grupo(pontuacoes, np.zeros((passos, series), dtype=int), alfa, minimo=30)
    por_grupo = conformal.conformal_por_grupo(pontuacoes, dificil.astype(int), alfa, minimo=30)
    validos = ~np.isnan(unico) & ~np.isnan(por_grupo)
    cobre_unico, cobre_grupo = pontuacoes <= unico, pontuacoes <= por_grupo
    # Na média de tudo, a calibração única entrega o prometido...
    assert cobre_unico[validos].mean() == pytest.approx(0.80, abs=0.02)
    # ...mas no grupo difícil fica muito abaixo. Valor teórico: P(|5Z| ≤ q), com q ≈ o quantil 80% da mistura.
    assert cobre_unico[validos & dificil].mean() < 0.30
    # Calibrando por grupo, cada grupo recebe o seu raio e cobre perto de 80%.
    assert cobre_grupo[validos & dificil].mean() == pytest.approx(0.80, abs=0.05)
    assert cobre_grupo[validos & ~dificil].mean() == pytest.approx(0.80, abs=0.02)
    # O raio do grupo difícil é cerca de cinco vezes o do fácil.
    assert np.nanmedian(por_grupo[dificil]) / np.nanmedian(por_grupo[~dificil]) == pytest.approx(5.0, rel=0.2)


def test_conformal_por_grupo_com_um_rotulo_por_serie_e_o_conformal_simples():
    gerador = np.random.default_rng(7)
    pontuacoes = np.abs(gerador.normal(size=(200, 4))) * np.array([1.0, 2.0, 5.0, 10.0])
    grupos = np.tile(np.arange(4), (200, 1))
    por_grupo = conformal.conformal_por_grupo(pontuacoes, grupos, 0.2, atraso=2, minimo=30)
    for s in range(4):
        np.testing.assert_array_equal(por_grupo[:, s], conformal.conformal_simples(pontuacoes[:, s], 0.2, atraso=2, minimo=30))
