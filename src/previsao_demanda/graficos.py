"""Figuras do estudo. A cor segue a entidade: azul é o método principal de cada figura, laranja é a
linha de base (ou o que o modelo entrega por padrão), verde-água é a alternativa, cinza é referência."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from previsao_demanda.estilo import (
    APAGADO,
    AQUA,
    AZUL,
    ESTILO,
    LARANJA,
    TINTA,
    TINTA_SECUNDARIA,
    milhar,
    nova_figura,
    numero,
    percentual,
)

NOMES_DOS_MODELOS = {
    "gbm": "Gradient boosting com calendário",
    "sazonal": "Sazonal ingênua (mesma semana do ano anterior)",
    "ets": "Suavização exponencial",
    "media_movel": "Média móvel de 4 semanas",
}
CORES_DOS_MODELOS = {"gbm": AZUL, "sazonal": LARANJA, "ets": AQUA, "media_movel": APAGADO}
NOMES_DOS_METODOS = {
    "parametrico": "Paramétrico do modelo",
    "conformal": "Conformal simples",
    "aci": "Conformal adaptativo (ACI)",
    "grupo": "Conformal por grupo",
}
CORES_DOS_METODOS = {"aci": AZUL, "parametrico": LARANJA, "conformal": AQUA}


def _eixo_percentual(eixo) -> None:
    eixo.set_major_formatter(FuncFormatter(lambda v, _: percentual(v)))


def cobertura_em_eventos(eventos: pd.DataFrame, nivel: float = 0.80) -> Figure:
    """Cobertura real nas semanas de data comemorativa contra as demais, para cada método."""
    metodos = list(NOMES_DOS_METODOS)
    simples = eventos.loc[("parametrico", nivel, "semana de evento", 0), "cobertura"]
    por_grupo = eventos.loc[("grupo", nivel, "semana de evento", 0), "cobertura"]
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            f"Na semana da data comemorativa, o intervalo de {percentual(nivel)} cobre só {percentual(simples)}; calibrado à parte, {percentual(por_grupo)}",
            f"Cobertura real do intervalo de {percentual(nivel)}: fração das previsões em que o valor ficou dentro da faixa. Barra: intervalo de 95% (bootstrap).",
            altura=3.9,
        )
        fig.subplots_adjust(left=0.25, bottom=0.25)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(nivel, color=TINTA_SECUNDARIA, linewidth=1.2, zorder=1)
        ax.annotate(f"prometido: {percentual(nivel)}", (nivel, -0.72), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        recortes = {"semana de evento": (AZUL, "Semana da data comemorativa e a anterior"), "demais semanas": (APAGADO, "Demais semanas")}
        for y, metodo in enumerate(metodos):
            for recorte, (cor, rotulo) in recortes.items():
                linha = eventos.loc[(metodo, nivel, recorte, 0)]
                ax.plot([linha["cobertura_inf"], linha["cobertura_sup"]], [y, y], color=cor, linewidth=2, zorder=2)
                ax.plot(linha["cobertura"], y, "o", color=cor, zorder=3, label=rotulo if y == 0 else None)
                if recorte == "semana de evento":
                    ax.annotate(percentual(linha["cobertura"]), (linha["cobertura"], y), xytext=(0, 9), textcoords="offset points", ha="center", color=TINTA, fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(metodos)), [NOMES_DOS_METODOS[m] for m in metodos])
        ax.set_ylim(-1.0, len(metodos) - 0.45)
        ax.invert_yaxis()
        ax.set_xlim(0.2, 1.0)
        _eixo_percentual(ax.xaxis)
        ax.set_xlabel("cobertura real")
        ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=9)
        return fig


def cobertura_por_horizonte(cobertura: pd.DataFrame) -> Figure:
    """Cobertura real por horizonte, nos dois níveis prometidos, um painel para cada."""
    cobertura = cobertura.sort_index()
    prometido, entregue = 0.95, cobertura.loc[("parametrico", 0.95, 0), "cobertura"]
    with plt.rc_context(ESTILO):
        fig, eixos = plt.subplots(1, 2, figsize=(8.2, 4.4), dpi=160)
        fig.subplots_adjust(top=0.76, left=0.08, right=0.97, bottom=0.23, wspace=0.22)
        fig.text(0.02, 0.955, f"No geral, o intervalo paramétrico promete {percentual(prometido)} e entrega {percentual(entregue)}; os conformes entregam", fontsize=13, fontweight="bold", color=TINTA, va="top")
        fig.text(0.02, 0.885, "Cobertura real por horizonte, juntando todas as semanas e os dez artigos. Faixa: intervalo de 95% (bootstrap em blocos).", fontsize=9.5, color=TINTA_SECUNDARIA, va="top")
        limites = {0.80: (0.70, 0.90), 0.95: (0.82, 1.02)}
        for ax, nivel in zip(eixos, (0.80, 0.95)):
            ax.axhline(nivel, color=TINTA_SECUNDARIA, linewidth=1.2, zorder=1)
            for metodo, cor in CORES_DOS_METODOS.items():
                tabela = cobertura.loc[(metodo, nivel)].drop(index=0)
                if metodo != "conformal":
                    ax.fill_between(tabela.index, tabela["cobertura_inf"], tabela["cobertura_sup"], color=cor, alpha=0.13, linewidth=0, zorder=2)
                ax.plot(tabela.index, tabela["cobertura"], "-o", color=cor, markersize=6, zorder=3, label=NOMES_DOS_METODOS[metodo])
            ax.set_title(f"Prometido: {percentual(nivel)}", fontsize=10, color=TINTA_SECUNDARIA, loc="left", pad=8)
            ax.set_ylim(*limites[nivel])
            ax.set_xticks(range(1, 9))
            _eixo_percentual(ax.yaxis)
            ax.set_xlabel("horizonte (semanas à frente)")
        eixos[0].set_ylabel("cobertura real")
        marcas, rotulos = eixos[0].get_legend_handles_labels()
        ordem = [rotulos.index(NOMES_DOS_METODOS[m]) for m in ("parametrico", "conformal", "aci")]
        fig.legend([marcas[i] for i in ordem], [rotulos[i] for i in ordem], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.01))
        return fig


def erro_por_horizonte(erros: pd.DataFrame, comparacao: pd.DataFrame) -> Figure:
    """MASE de cada previsor por horizonte, com a faixa de incerteza do modelo principal."""
    reducao = 1 - comparacao.loc["gbm", "razao_para_a_sazonal"]
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            f"O gradient boosting com calendário erra {percentual(reducao)} menos que repetir o ano anterior",
            "MASE por horizonte (erro absoluto dividido pelo erro histórico da sazonal ingênua; menor é melhor). Faixa: intervalo de 95%.",
        )
        fig.subplots_adjust(bottom=0.27)
        for modelo in ("media_movel", "sazonal", "ets", "gbm"):
            tabela = erros.loc[modelo]
            cor = CORES_DOS_MODELOS[modelo]
            if modelo == "gbm":
                ax.fill_between(tabela.index, tabela["mase_inf"], tabela["mase_sup"], color=cor, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(tabela.index, tabela["mase"], "-o", color=cor, markersize=6, zorder=4 if modelo == "gbm" else 3, label=NOMES_DOS_MODELOS[modelo])
        for modelo, deslocamento in (("gbm", (0, -13)), ("sazonal", (0, 8))):
            for h in (1, 8):
                valor = erros.loc[(modelo, h), "mase"]
                ax.annotate(numero(valor), (h, valor), xytext=deslocamento, textcoords="offset points", ha="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        teto = float(np.ceil(erros["mase"].max() * 2) / 2)
        ax.set_ylim(0, teto)
        ax.set_yticks(np.arange(0, teto + 0.01, 0.5))
        ax.set_xticks(range(1, 9))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1)))
        ax.set_xlabel("horizonte (semanas à frente)")
        ax.set_ylabel("MASE")
        marcas, rotulos = ax.get_legend_handles_labels()
        ax.legend(marcas[::-1], rotulos[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)
        return fig


def erro_por_artigo(por_artigo: pd.DataFrame) -> Figure:
    """Razão entre o erro do gradient boosting e o da sazonal ingênua, artigo por artigo."""
    tabela = por_artigo.sort_values("razao_gbm")
    vence = int((tabela["razao_gbm_sup"] < 1).sum())
    empata = int(((tabela["razao_gbm_inf"] <= 1) & (tabela["razao_gbm_sup"] >= 1)).sum())
    final = f"em {empata} é empate" if empata else "em nenhum empata"
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            f"O modelo erra menos que a sazonal ingênua em {vence} dos {len(tabela)} artigos; {final}",
            "Erro do gradient boosting dividido pelo da sazonal ingênua (média dos 8 horizontes). Barra: intervalo de 95% (bootstrap pareado).",
            altura=4.3,
        )
        fig.subplots_adjust(left=0.2, bottom=0.22)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(1, color=TINTA_SECUNDARIA, linewidth=1.2, zorder=1)
        ax.annotate("erro igual ao da sazonal ingênua", (1, -0.75), xytext=(5, 0), textcoords="offset points", va="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        rotulado = set()
        for y, (artigo, linha) in enumerate(tabela.iterrows()):
            ganha = linha["razao_gbm_sup"] < 1
            cor, rotulo = (AZUL, "Erra menos (intervalo inteiro abaixo de 1)") if ganha else (APAGADO, "Empate (intervalo contém 1)")
            ax.plot([linha["razao_gbm_inf"], linha["razao_gbm_sup"]], [y, y], color=cor, linewidth=2, zorder=2)
            ax.plot(linha["razao_gbm"], y, "o", color=cor, zorder=3, label=None if rotulo in rotulado else rotulo)
            rotulado.add(rotulo)
            if y in (0, len(tabela) - 1):  # rótulo só nos extremos; os demais valores estão na tabela do README
                ax.annotate(numero(linha["razao_gbm"]), (linha["razao_gbm"], y), xytext=(0, 8), textcoords="offset points", ha="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        ax.set_yticks(range(len(tabela)), list(tabela.index))
        ax.set_ylim(-1.1, len(tabela) - 0.4)
        ax.invert_yaxis()
        ax.set_xlim(0.2, max(1.5, float(tabela["razao_gbm_sup"].max()) + 0.05))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1)))
        ax.set_xlabel("erro do modelo ÷ erro da sazonal ingênua (abaixo de 1, o modelo erra menos)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.19), ncol=2, fontsize=9)
        return fig


def exemplo_de_intervalos(exemplo: pd.DataFrame, artigo: str, h: int, desde: str = "2023-07-01") -> Figure:
    """Um artigo, um horizonte: o realizado, a previsão e as duas faixas de 80%."""
    tabela = exemplo[exemplo.index >= desde]
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            "A faixa calibrada por grupo se abre na semana do evento, que é onde o erro mora",
            f"{artigo}: previsão feita {h} semanas antes (suavização exponencial) e intervalos de 80%. Escala logarítmica.",
        )
        fig.subplots_adjust(bottom=0.25)
        ax.fill_between(tabela.index, tabela["grupo_inf"], tabela["grupo_sup"], color=AZUL, alpha=0.18, linewidth=0, zorder=2, label="Intervalo conformal por grupo")
        ax.plot(tabela.index, tabela["conformal_inf"], color=LARANJA, linewidth=1.2, zorder=3, label="Limites do conformal simples")
        ax.plot(tabela.index, tabela["conformal_sup"], color=LARANJA, linewidth=1.2, zorder=3)
        ax.plot(tabela.index, tabela["previsto"], color=AZUL, linewidth=1.6, zorder=4, label="Previsão")
        ax.plot(tabela.index, tabela["real"], color=TINTA, linewidth=1.3, zorder=5, label="Realizado")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: milhar(v)))
        ax.minorticks_off()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        ax.set_ylabel("visualizações na semana")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=9)
        return fig


def cobertura_por_ano(por_ano: pd.DataFrame, nivel: float = 0.80) -> Figure:
    """Cobertura real em cada ano: quem acompanha quando os erros mudam de tamanho."""
    ultimo = int(por_ano.index[-1])
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            f"Quando os erros cresceram, em {ultimo - 1} e {ultimo}, só o adaptativo ficou perto dos {percentual(nivel)}",
            f"Cobertura real do intervalo de {percentual(nivel)} por ano em que a previsão foi feita ({int(por_ano.index[0])} e {ultimo} são anos parciais).",
        )
        fig.subplots_adjust(bottom=0.25)
        ax.axhline(nivel, color=TINTA_SECUNDARIA, linewidth=1.2, zorder=1)
        ax.annotate(f"prometido: {percentual(nivel)}", (ultimo - 1.5, nivel), xytext=(0, 5), textcoords="offset points", ha="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        for metodo in ("parametrico", "conformal", "aci"):
            ax.plot(por_ano.index, por_ano[metodo], "-o", color=CORES_DOS_METODOS[metodo], markersize=6, zorder=4 if metodo == "aci" else 3, label=NOMES_DOS_METODOS[metodo])
        ax.annotate(percentual(por_ano["aci"].iloc[-1]), (ultimo, por_ano["aci"].iloc[-1]), xytext=(0, 8), textcoords="offset points", ha="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        ax.annotate(percentual(por_ano["conformal"].iloc[-1]), (ultimo, por_ano["conformal"].iloc[-1]), xytext=(0, -14), textcoords="offset points", ha="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        ax.set_ylim(0.66, 0.90)
        ax.set_xticks(list(por_ano.index))
        _eixo_percentual(ax.yaxis)
        ax.set_xlabel("ano em que a previsão foi feita")
        ax.set_ylabel("cobertura real")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9)
        return fig
