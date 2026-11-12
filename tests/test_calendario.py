from datetime import date

import numpy as np
import pandas as pd
import pytest

from previsao_demanda import calendario


@pytest.mark.parametrize(
    ("ano", "esperado"),
    [(2011, date(2011, 4, 24)), (2016, date(2016, 3, 27)), (2019, date(2019, 4, 21)), (2024, date(2024, 3, 31)),
     (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5)), (2038, date(2038, 4, 25))],
)
def test_pascoa_bate_com_datas_conhecidas(ano, esperado):
    assert calendario.pascoa(ano) == esperado
    assert calendario.pascoa(ano).weekday() == 6  # sempre domingo


@pytest.mark.parametrize(("ano", "esperado"), [(2020, date(2020, 2, 25)), (2024, date(2024, 2, 13)), (2025, date(2025, 3, 4)), (2026, date(2026, 2, 17))])
def test_carnaval_e_a_terca_47_dias_antes_da_pascoa(ano, esperado):
    assert calendario.carnaval(ano) == esperado
    assert calendario.carnaval(ano).weekday() == 1


def test_datas_moveis_do_comercio():
    assert calendario.dia_das_maes(2023) == date(2023, 5, 14)
    assert calendario.dia_das_maes(2024) == date(2024, 5, 12)
    assert calendario.dia_das_maes(2026) == date(2026, 5, 10)
    assert calendario.dia_dos_pais(2025) == date(2025, 8, 10)
    assert calendario.dia_dos_pais(2026) == date(2026, 8, 9)
    assert calendario.black_friday(2019) == date(2019, 11, 29)
    assert calendario.black_friday(2024) == date(2024, 11, 29)
    assert calendario.black_friday(2025) == date(2025, 11, 28)
    assert calendario.black_friday(2026) == date(2026, 11, 27)
    for ano in range(2015, 2030):
        assert calendario.black_friday(ano).weekday() == 4
        assert calendario.dia_das_maes(ano).weekday() == 6 and 8 <= calendario.dia_das_maes(ano).day <= 14


def test_semanas_ate_evento_tem_sinal_e_zero_na_semana_do_evento():
    # A Páscoa de 2025 foi no domingo 20/04: a semana (segunda a domingo) começa em 14/04.
    semanas = pd.DatetimeIndex(["2025-03-31", "2025-04-07", "2025-04-14", "2025-04-21", "2025-12-22"])
    # Em 22/12/2025 a Páscoa mais próxima já é a de 2026 (05/04, semana de 30/03): faltam 14 semanas.
    np.testing.assert_array_equal(calendario.semanas_ate_evento(semanas, "pascoa"), [2, 1, 0, -1, 14])
    # Natal de 2025 cai na quinta 25/12, dentro da semana de 22/12.
    assert calendario.semanas_ate_evento(semanas, "natal")[-1] == 0
    # O Carnaval fica sempre seis semanas antes da semana da Páscoa.
    todas = pd.date_range("2015-07-06", periods=600, freq="7D")
    np.testing.assert_array_equal(
        calendario.semanas_ate_evento(todas, "carnaval")[calendario.semanas_ate_evento(todas, "pascoa") == 0], -6
    )


def test_semana_alinhada_aponta_para_a_mesma_distancia_do_evento_no_ano_anterior():
    inicio = pd.Timestamp("2015-07-06")
    semanas = pd.date_range(inicio, periods=600, freq="7D")
    posicao = calendario.semana_alinhada_do_ano_anterior(semanas, "pascoa", inicio)
    distancia = calendario.semanas_ate_evento(semanas, "pascoa")
    # Semana da Páscoa de 2025 (14/04/2025) alinha com a da Páscoa de 2024 (25/03/2024): 55 semanas antes, não 52.
    i = semanas.get_loc(pd.Timestamp("2025-04-14"))
    assert semanas[posicao[i]] == pd.Timestamp("2024-03-25")
    assert i - posicao[i] == 55
    # Para toda semana com alinhada dentro da série e perto do evento, a distância ao evento é a mesma.
    perto = (posicao >= 0) & (np.abs(distancia) <= 6)
    np.testing.assert_array_equal(distancia[posicao[perto]], distancia[perto])
    # A semana alinhada está sempre no passado, a cerca de um ano.
    assert ((np.arange(600) - posicao)[perto] >= 47).all() and ((np.arange(600) - posicao)[perto] <= 57).all()
    # No primeiro ano da série não existe semana alinhada.
    assert (posicao[:40] == -1).all()
