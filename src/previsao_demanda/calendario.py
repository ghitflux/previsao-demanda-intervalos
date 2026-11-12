"""Datas comemorativas calculadas no código, e a posição de cada semana em relação a elas.

Datas móveis mudam de semana de um ano para o outro. "A mesma semana do ano passado" erra o pico
quando a Páscoa cai três semanas mais cedo; por isso o modelo recebe a distância até o evento.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd


def pascoa(ano: int) -> date:
    """Domingo de Páscoa no calendário gregoriano (algoritmo de Meeus/Jones/Butcher)."""
    a = ano % 19
    b, c = divmod(ano, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    mes, dia = divmod(h + m - 7 * n + 114, 31)
    return date(ano, mes, dia + 1)


def carnaval(ano: int) -> date:
    """Terça-feira de Carnaval: 47 dias antes do domingo de Páscoa."""
    return pascoa(ano) - timedelta(days=47)


def _enesimo_dia_da_semana(ano: int, mes: int, dia_da_semana: int, n: int) -> date:
    """O n-ésimo dia da semana (0 = segunda, 6 = domingo) do mês."""
    primeiro = date(ano, mes, 1)
    avanco = (dia_da_semana - primeiro.weekday()) % 7
    return primeiro + timedelta(days=avanco + 7 * (n - 1))


def dia_das_maes(ano: int) -> date:
    """Segundo domingo de maio."""
    return _enesimo_dia_da_semana(ano, 5, 6, 2)


def dia_dos_pais(ano: int) -> date:
    """Segundo domingo de agosto (data brasileira)."""
    return _enesimo_dia_da_semana(ano, 8, 6, 2)


def black_friday(ano: int) -> date:
    """Sexta-feira seguinte à quarta quinta-feira de novembro (o Dia de Ação de Graças dos EUA)."""
    return _enesimo_dia_da_semana(ano, 11, 3, 4) + timedelta(days=1)


def natal(ano: int) -> date:
    return date(ano, 12, 25)


def dia_dos_namorados(ano: int) -> date:
    """12 de junho (data brasileira)."""
    return date(ano, 6, 12)


def sao_joao(ano: int) -> date:
    """24 de junho, o centro das festas juninas."""
    return date(ano, 6, 24)


def sao_valentim(ano: int) -> date:
    """14 de fevereiro: o Dia dos Namorados de Portugal e de boa parte do mundo."""
    return date(ano, 2, 14)


EVENTOS: dict[str, Callable[[int], date]] = {
    "sao_valentim": sao_valentim,
    "carnaval": carnaval,
    "pascoa": pascoa,
    "dia_das_maes": dia_das_maes,
    "dia_dos_namorados": dia_dos_namorados,
    "sao_joao": sao_joao,
    "dia_dos_pais": dia_dos_pais,
    "black_friday": black_friday,
    "natal": natal,
}
MOVEIS = ["carnaval", "pascoa", "dia_das_maes", "dia_dos_pais", "black_friday"]


def segunda_da_semana(dia: date) -> date:
    return dia - timedelta(days=dia.weekday())


def semanas_ate_evento(semanas: pd.DatetimeIndex, evento: str) -> np.ndarray:
    """Distância, em semanas, de cada semana até a semana da ocorrência mais próxima do evento.

    Positivo: o evento ainda vai acontecer. Zero: o evento cai nesta semana. Negativo: já passou.
    `semanas` são segundas-feiras.
    """
    funcao = EVENTOS[evento]
    anos = range(semanas.year.min() - 1, semanas.year.max() + 2)
    ocorrencias = np.array([pd.Timestamp(segunda_da_semana(funcao(ano))).to_datetime64() for ano in anos])
    diferenca = (ocorrencias[None, :] - semanas.to_numpy()[:, None]) / np.timedelta64(7, "D")
    mais_proxima = np.abs(diferenca).argmin(axis=1)
    return np.round(diferenca[np.arange(len(semanas)), mais_proxima]).astype(int)


def semana_alinhada_do_ano_anterior(semanas: pd.DatetimeIndex, evento: str, inicio: pd.Timestamp) -> np.ndarray:
    """Posição da semana do ano anterior que ficava à mesma distância do evento.

    Exemplo: a semana que antecede a Páscoa de 2025 é alinhada com a semana que antecedeu a
    Páscoa de 2024, mesmo que as duas estejam a 55 semanas uma da outra. A posição é contada em
    semanas a partir de `inicio` (a primeira semana da série); devolve -1 quando a semana
    alinhada cai antes dele.
    """
    funcao = EVENTOS[evento]
    anos = list(range(semanas.year.min() - 2, semanas.year.max() + 2))
    ocorrencias = np.array([pd.Timestamp(segunda_da_semana(funcao(ano))).to_datetime64() for ano in anos])
    diferenca = (ocorrencias[None, :] - semanas.to_numpy()[:, None]) / np.timedelta64(7, "D")
    mais_proxima = np.abs(diferenca).argmin(axis=1)
    distancia = np.round(diferenca[np.arange(len(semanas)), mais_proxima]).astype(int)
    anterior = ocorrencias[mais_proxima - 1]  # `anos` começa dois anos antes: sempre existe a ocorrência anterior
    alvo = anterior - distancia * np.timedelta64(7, "D")
    posicao = np.round((alvo - inicio.to_datetime64()) / np.timedelta64(7, "D")).astype(int)
    posicao[posicao < 0] = -1
    return posicao
