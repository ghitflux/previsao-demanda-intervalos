"""Carga e agregação das visualizações de artigos da Wikipédia em português.

Visualização de artigo é um indicador de interesse, não de venda: mede quanta gente foi ler sobre
o assunto naquela semana. Serve aqui como série pública, longa e sazonal para testar o método.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DADOS = Path("data")
ARQUIVO_SEMANAL = "visualizacoes_semanais.csv"
ARQUIVO_HASH = "visualizacoes_semanais.sha256"
INICIO, FIM = "2015-07-01", "2026-10-31"

# Título do artigo na Wikipédia (como a API espera) → rótulo de exibição e eventos de calendário ligados a ele.
# Sem evento: a data muda por edital ou por lei a cada ano e não pode ser calculada no código.
# O artigo "Dia dos Namorados" cobre as duas datas: 14 de fevereiro (São Valentim) e 12 de junho (Brasil).
ARTIGOS: dict[str, dict] = {
    "Black_Friday": {"rotulo": "Black Friday", "eventos": ["black_friday"]},
    "Dia_das_Mães": {"rotulo": "Dia das Mães", "eventos": ["dia_das_maes"]},
    "Dia_dos_Pais": {"rotulo": "Dia dos Pais", "eventos": ["dia_dos_pais"]},
    "Dia_dos_Namorados": {"rotulo": "Dia dos Namorados", "eventos": ["sao_valentim", "dia_dos_namorados"]},
    "Natal": {"rotulo": "Natal", "eventos": ["natal"]},
    "Páscoa": {"rotulo": "Páscoa", "eventos": ["pascoa"]},
    "Carnaval": {"rotulo": "Carnaval", "eventos": ["carnaval"]},
    "Festa_junina": {"rotulo": "Festa junina", "eventos": ["sao_joao"]},
    "Imposto_de_renda": {"rotulo": "Imposto de renda", "eventos": []},
    "Exame_Nacional_do_Ensino_Médio": {"rotulo": "Enem", "eventos": []},
}


def rotulo(artigo: str) -> str:
    return str(ARTIGOS[artigo]["rotulo"])


def agregar_semanal(diario: pd.DataFrame) -> pd.DataFrame:
    """Soma as visualizações diárias em semanas de segunda a domingo, só com semanas completas.

    `diario` tem índice de datas e uma coluna por artigo. Semana com menos de 7 dias (as pontas do
    período) fica de fora: somar 3 dias e chamar de semana criaria uma queda que não existe.
    O índice do resultado é a segunda-feira de cada semana.
    """
    diario = diario.sort_index()
    segunda = diario.index - pd.to_timedelta(diario.index.day_of_week, unit="D")
    dias = diario.groupby(segunda).size()
    semanal = diario.groupby(segunda).sum()
    semanal = semanal[dias == 7]
    semanal.index.name = "semana"
    return semanal.astype("int64")


def texto_csv(semanal: pd.DataFrame) -> str:
    """O extrato semanal como texto, sempre com fim de linha LF (o hash não pode depender do sistema)."""
    saida = semanal.copy()
    saida.index = saida.index.strftime("%Y-%m-%d")
    return saida.to_csv(lineterminator="\n")


def hash_do_extrato(texto: str) -> str:
    return hashlib.sha256(texto.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def carregar_semanal(pasta: Path = DADOS) -> pd.DataFrame:
    """Extrato versionado: uma linha por semana (segunda-feira), uma coluna por artigo."""
    semanal = pd.read_csv(pasta / ARQUIVO_SEMANAL, index_col="semana", parse_dates=True)
    return semanal[list(ARTIGOS)]
