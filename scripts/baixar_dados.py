"""Baixa as visualizações diárias da API pública da Wikimedia e grava o extrato semanal em data/.

Uma requisição por artigo, uma de cada vez, com pausa entre elas e User-Agent genérico. Os
arquivos brutos (diários) ficam em data/brutos/, fora do git; o extrato semanal é pequeno e é
versionado, então a análise roda sem rede. Ao final, confere o tamanho da série e o hash do
extrato contra o que está versionado.

Os títulos foram conferidos em 11/11/2026: cada um é o título canônico do artigo (não é
redirecionamento) e tem todos os dias do período. "Dia das Crianças" e "Dia das Bruxas" foram
descartados por terem dias sem registro; "Carnaval do Brasil", por repetir o tema de "Carnaval".
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from previsao_demanda import dados

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/pt.wikipedia/all-access/user/{artigo}/daily/{inicio}/{fim}"
AGENTE = {"User-Agent": "portfolio-estudos/1.0"}
PAUSA_SEGUNDOS = 2.0
BRUTOS = dados.DADOS / "brutos"


def baixar(artigo: str) -> list[dict]:
    """Série diária de um artigo. Em caso de limite de requisições (429), espera e tenta de novo."""
    url = API.format(
        artigo=urllib.parse.quote(artigo, safe=""),
        inicio=dados.INICIO.replace("-", ""),
        fim=dados.FIM.replace("-", ""),
    )
    for tentativa in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=60) as resposta:
                return json.loads(resposta.read().decode("utf-8"))["items"]
        except urllib.error.HTTPError as erro:
            if erro.code != 429:
                raise
            espera = min(float(erro.headers.get("Retry-After") or 15 * (tentativa + 1)), 90.0)
            print(f"  limite de requisições; esperando {espera:.0f} s")
            time.sleep(espera)
    raise RuntimeError(f"Não foi possível baixar {artigo}: limite de requisições.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    BRUTOS.mkdir(parents=True, exist_ok=True)
    dias_esperados = len(pd.date_range(dados.INICIO, dados.FIM, freq="D"))

    series = {}
    for artigo in dados.ARTIGOS:
        itens = baixar(artigo)
        (BRUTOS / f"{artigo}.json").write_text(json.dumps(itens, ensure_ascii=False), encoding="utf-8")
        serie = pd.Series(
            [item["views"] for item in itens],
            index=pd.to_datetime([item["timestamp"][:8] for item in itens], format="%Y%m%d"),
        )
        if len(serie) != dias_esperados or serie.index.has_duplicates:
            raise SystemExit(f"{artigo}: esperados {dias_esperados} dias, vieram {len(serie)}. Série incompleta.")
        series[artigo] = serie
        total = f"{int(serie.sum()):,d}".replace(",", ".")
        print(f"{artigo:34s} {len(serie)} dias, {total:>10s} visualizações")
        time.sleep(PAUSA_SEGUNDOS)

    semanal = dados.agregar_semanal(pd.DataFrame(series))
    texto = dados.texto_csv(semanal)
    novo_hash = dados.hash_do_extrato(texto)

    arquivo_hash = dados.DADOS / dados.ARQUIVO_HASH
    if arquivo_hash.exists():
        versionado = arquivo_hash.read_text(encoding="utf-8").strip()
        if versionado == novo_hash:
            print("Hash confere com o extrato versionado.")
        else:
            print("ATENÇÃO: o hash mudou em relação ao extrato versionado (a Wikimedia pode ter revisado a série).")
            print(f"  versionado: {versionado}\n  novo:       {novo_hash}")
    (dados.DADOS / dados.ARQUIVO_SEMANAL).write_text(texto, encoding="utf-8", newline="\n")
    arquivo_hash.write_text(novo_hash + "\n", encoding="utf-8", newline="\n")
    print(f"{len(semanal)} semanas completas, de {semanal.index[0]:%d/%m/%Y} a {semanal.index[-1]:%d/%m/%Y} (segundas-feiras).")
    print(f"Gravado em {dados.DADOS / dados.ARQUIVO_SEMANAL} (sha256 {novo_hash[:16]}…)")


if __name__ == "__main__":
    main()
