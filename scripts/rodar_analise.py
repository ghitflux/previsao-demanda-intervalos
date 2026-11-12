"""Roda a análise completa e grava reports/resultados.json e as figuras do README."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from previsao_demanda import analise, dados, graficos, previsores, resumo_site

SAIDA = Path("reports")
FIGURAS = SAIDA / "figuras"


def _registros(tabela: pd.DataFrame, casas: int = 6) -> list[dict]:
    """Tabela como lista de registros, com o índice virando colunas."""
    return json.loads(tabela.reset_index().round(casas).to_json(orient="records", force_ascii=False))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    FIGURAS.mkdir(parents=True, exist_ok=True)

    semanal = dados.carregar_semanal()
    r = analise.rodar(semanal)

    figuras = {
        "cobertura_em_eventos": graficos.cobertura_em_eventos(r.cobertura_em_eventos),
        "cobertura_por_horizonte": graficos.cobertura_por_horizonte(r.cobertura),
        "erro_por_horizonte": graficos.erro_por_horizonte(r.erro_por_horizonte, r.comparacao),
        "erro_por_artigo": graficos.erro_por_artigo(r.erro_por_artigo),
        "exemplo_de_intervalos": graficos.exemplo_de_intervalos(
            r.exemplo, dados.rotulo(analise.ARTIGO_DE_EXEMPLO), analise.HORIZONTE_DE_EXEMPLO
        ),
        "cobertura_por_ano": graficos.cobertura_por_ano(r.cobertura_por_ano),
    }
    for nome, figura in figuras.items():
        figura.savefig(FIGURAS / f"{nome}.png")
        plt.close(figura)

    resultados = {
        "base": r.base,
        "configuracao": {
            "horizonte_max": analise.HORIZONTE_MAX,
            "primeira_origem": analise.PRIMEIRA_ORIGEM,
            "origens_de_aquecimento": analise.AQUECIMENTO,
            "niveis": list(analise.NIVEIS),
            "gama_da_aci": analise.GAMA,
            "minimo_de_calibracao": analise.MINIMO_CALIBRACAO,
            "semanas_de_evento": list(analise.SEMANAS_DE_EVENTO),
            "bloco_do_bootstrap": analise.BLOCO,
            "repeticoes_do_bootstrap": analise.REPETICOES,
            "semente": analise.SEMENTE,
            "reajuste_dos_modelos": previsores.REAJUSTE,
            "parametros_do_gradient_boosting": {**previsores.PARAMETROS_GBM, "arvores": previsores.ARVORES_GBM},
            "artigos": {artigo: info for artigo, info in dados.ARTIGOS.items()},
        },
        "achados": r.achados,
        "comparacao_com_a_linha_de_base": _registros(r.comparacao),
        "erro_por_horizonte": _registros(r.erro_por_horizonte),
        "razao_por_horizonte": {modelo: _registros(tabela) for modelo, tabela in r.razao_por_horizonte.items()},
        "erro_por_artigo": _registros(r.erro_por_artigo),
        "cobertura_dos_intervalos": _registros(r.cobertura),
        "cobertura_por_ano": _registros(r.cobertura_por_ano),
        "cobertura_em_eventos": _registros(r.cobertura_em_eventos),
        "conformal_por_modelo": _registros(r.conformal_por_modelo),
        "sensibilidade_ao_gama": _registros(r.sensibilidade_ao_gama),
        "sensibilidade_a_folha": r.sensibilidade_a_folha,
        "para_o_site": resumo_site.para_o_site(r),
    }
    (SAIDA / "resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8", newline="\n"
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(r.base)
    print("\nComparação com a linha de base (média dos horizontes):")
    print(r.comparacao.drop(columns=["nome"]).to_string())
    print("\nMASE por horizonte:")
    print(r.erro_por_horizonte["mase"].unstack("horizonte").to_string())
    print("\nErro por artigo:")
    print(r.erro_por_artigo[["mase_sazonal", "mase_ets", "mase_gbm", "razao_gbm", "razao_gbm_inf", "razao_gbm_sup"]].to_string())
    print("\nCobertura dos intervalos (todas as semanas, todos os horizontes):")
    print(r.cobertura.xs(0, level="horizonte").to_string())
    print("\nCobertura em semanas de evento e nas demais (todos os horizontes):")
    print(r.cobertura_em_eventos.xs(0, level="horizonte").to_string())
    print("\nCobertura de 80% por ano:")
    print(r.cobertura_por_ano.to_string())
    print("\nConformal em volta de cada previsor (80%):")
    print(r.conformal_por_modelo.drop(columns=["nome"]).to_string())
    print("\nSensibilidade da ACI ao passo de correção:")
    print(r.sensibilidade_ao_gama.to_string())
    print("\nGradient boosting com o tamanho de folha do primeiro desenho:", r.sensibilidade_a_folha)
    print("\nAchados:", json.dumps(r.achados, indent=1, ensure_ascii=False))
    print(f"\nGravado em {SAIDA / 'resultados.json'} e {FIGURAS}")
    print(f"Tempo total: {time.perf_counter() - inicio:.0f} s de relógio, {time.process_time():.0f} s de processador")


if __name__ == "__main__":
    main()
