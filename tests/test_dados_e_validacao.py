from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from previsao_demanda import dados, validacao

RAIZ = Path(__file__).resolve().parents[1]


def test_agregar_semanal_soma_de_segunda_a_domingo_e_descarta_semana_incompleta():
    # 01/07/2015 é quarta-feira: a primeira semana completa começa na segunda 06/07.
    dias = pd.date_range("2015-07-01", "2015-07-22", freq="D")
    diario = pd.DataFrame({"a": np.arange(len(dias)), "b": 1}, index=dias)
    semanal = dados.agregar_semanal(diario)
    assert list(semanal.index) == [pd.Timestamp("2015-07-06"), pd.Timestamp("2015-07-13")]
    assert (semanal.index.day_of_week == 0).all()
    assert list(semanal["b"]) == [7, 7]
    # 06/07 é o dia de índice 5: a semana soma 5 + 6 + … + 11 = 56.
    assert semanal["a"].iloc[0] == 56


def test_hash_do_extrato_nao_depende_do_fim_de_linha():
    semanal = pd.DataFrame({"a": [1, 2]}, index=pd.DatetimeIndex(["2020-01-06", "2020-01-13"], name="semana"))
    texto = dados.texto_csv(semanal)
    assert texto == "semana,a\n2020-01-06,1\n2020-01-13,2\n"
    assert dados.hash_do_extrato(texto) == dados.hash_do_extrato(texto.replace("\n", "\r\n"))


def test_extrato_versionado_esta_integro():
    """O arquivo que a análise lê é o mesmo que o script de download gravou."""
    pasta = RAIZ / "data"
    texto = (pasta / dados.ARQUIVO_SEMANAL).read_text(encoding="utf-8")
    assert dados.hash_do_extrato(texto) == (pasta / dados.ARQUIVO_HASH).read_text(encoding="utf-8").strip()
    semanal = dados.carregar_semanal(pasta)
    assert list(semanal.columns) == list(dados.ARTIGOS)
    assert (semanal.index.day_of_week == 0).all()
    assert (np.diff(semanal.index.to_numpy()) == np.timedelta64(7, "D")).all()  # nenhuma semana faltando
    assert (semanal > 0).all().all()


def test_origens_moveis_param_onde_ainda_ha_futuro_observado():
    origens = validacao.origens_moveis(n_semanas=100, primeira=60, horizonte_max=8)
    assert origens[0] == 60 and origens[-1] == 92  # 92 + 8 − 1 = 99, a última semana
    with pytest.raises(ValueError):
        validacao.origens_moveis(n_semanas=60, primeira=60, horizonte_max=8)


def test_realizado_alinha_origem_e_horizonte():
    y = np.arange(100.0)[:, None] * np.array([1.0, 10.0])
    real = validacao.realizado(y, np.array([60, 92]), 8)
    assert real.shape == (2, 8, 2)
    # Origem 60: horizonte 1 é a semana 60; horizonte 8 é a semana 67.
    assert real[0, 0, 0] == 60 and real[0, 7, 0] == 67 and real[1, 7, 1] == 990
