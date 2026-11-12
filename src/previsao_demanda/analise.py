"""A análise do estudo: backtest por origem móvel, erro por horizonte e cobertura dos intervalos.

Cada tabela do README e do notebook sai de uma função daqui. As constantes abaixo foram fixadas
antes de olhar a janela de avaliação.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from previsao_demanda import calendario, conformal, dados, metricas, previsores, validacao

HORIZONTE_MAX = 8  # semanas à frente
PRIMEIRA_ORIGEM = 156  # três anos de história antes da primeira previsão
AQUECIMENTO = 104  # origens que só juntam erros para calibrar os intervalos; não entram em nenhuma métrica
NIVEIS = (0.80, 0.95)  # coberturas nominais comparadas
GAMA = 0.02  # passo de correção da ACI
GAMAS_TESTADOS = (0.005, 0.01, 0.02, 0.05)
MINIMO_CALIBRACAO = 30  # erros conhecidos antes de emitir o primeiro intervalo conformal
BLOCO = 13  # origens por bloco no bootstrap (um trimestre)
REPETICOES = 2000
SEMENTE = 42
SEMANAS_DE_EVENTO = (0, 1)  # a semana da data comemorativa e a anterior

MODELOS = {
    "sazonal": "Sazonal ingênua",
    "media_movel": "Média móvel",
    "ets": "Suavização exponencial",
    "gbm": "Gradient boosting",
}
METODOS = {
    "parametrico": "Paramétrico do modelo",
    "conformal": "Conformal simples",
    "aci": "Conformal adaptativo (ACI)",
    "grupo": "Conformal por grupo",
}


@dataclass
class Backtest:
    """Tudo o que o backtest produz. Os arrays de previsão têm forma (origens, horizontes, séries)."""

    semanas: pd.DatetimeIndex
    artigos: list[str]
    y: np.ndarray
    origens: np.ndarray
    real: np.ndarray
    previsoes: dict[str, np.ndarray]
    desvio_ets: np.ndarray
    escalas: np.ndarray
    avaliadas: np.ndarray  # máscara das origens que entram nas métricas (depois do aquecimento)
    indices: np.ndarray  # reamostragens em blocos das origens avaliadas

    @property
    def horizontes(self) -> np.ndarray:
        return np.arange(1, self.real.shape[1] + 1)


def rodar_backtest(semanal: pd.DataFrame) -> Backtest:
    """Roda os quatro previsores em todas as origens, de três anos de história até o fim da série."""
    y = semanal.to_numpy(dtype=float)
    origens = validacao.origens_moveis(len(y), PRIMEIRA_ORIGEM, HORIZONTE_MAX)
    ets, desvio = previsores.suavizacao_exponencial(y, origens, HORIZONTE_MAX)
    previsoes = {
        "sazonal": previsores.sazonal_ingenua(y, origens, HORIZONTE_MAX),
        "media_movel": previsores.media_movel(y, origens, HORIZONTE_MAX),
        "ets": ets,
        "gbm": previsores.gradient_boosting(y, semanal.index, origens, HORIZONTE_MAX),
    }
    inicio_da_avaliacao = PRIMEIRA_ORIGEM + AQUECIMENTO
    avaliadas = origens >= inicio_da_avaliacao
    # Escala do MASE: erro da sazonal ingênua nas semanas anteriores à janela de avaliação.
    escalas = np.array([metricas.escala_mase(y[:inicio_da_avaliacao, s]) for s in range(y.shape[1])])
    return Backtest(
        semanas=semanal.index,
        artigos=list(semanal.columns),
        y=y,
        origens=origens,
        real=validacao.realizado(y, origens, HORIZONTE_MAX),
        previsoes=previsoes,
        desvio_ets=desvio,
        escalas=escalas,
        avaliadas=avaliadas,
        indices=metricas.indices_em_blocos(int(avaliadas.sum()), BLOCO, REPETICOES, SEMENTE),
    )


def resumo_da_base(bt: Backtest) -> dict:
    primeira, ultima = bt.origens[bt.avaliadas][0], bt.origens[bt.avaliadas][-1]
    return {
        "artigos": len(bt.artigos),
        "semanas": len(bt.semanas),
        "primeira_semana": f"{bt.semanas[0]:%Y-%m-%d}",
        "ultima_semana": f"{bt.semanas[-1]:%Y-%m-%d}",
        "visualizacoes_no_periodo": int(bt.y.sum()),
        "origens_no_backtest": len(bt.origens),
        "origens_de_aquecimento": int((~bt.avaliadas).sum()),
        "origens_avaliadas": int(bt.avaliadas.sum()),
        "primeira_origem_avaliada": f"{bt.semanas[primeira]:%Y-%m-%d}",
        "ultima_origem_avaliada": f"{bt.semanas[ultima]:%Y-%m-%d}",
        "previsoes_avaliadas_por_modelo": int(bt.avaliadas.sum()) * HORIZONTE_MAX * len(bt.artigos),
        "horizonte_max": HORIZONTE_MAX,
        "bloco_do_bootstrap": BLOCO,
        "repeticoes_do_bootstrap": REPETICOES,
        "reajuste_dos_modelos_em_semanas": previsores.REAJUSTE,
    }


# ---------------------------------------------------------------- erro das previsões pontuais


def _erros(bt: Backtest, modelo: str) -> np.ndarray:
    """Erro absoluto na escala original, só nas origens avaliadas: (origens, horizontes, séries)."""
    return np.abs(bt.previsoes[modelo] - bt.real)[bt.avaliadas]


def erro_por_horizonte(bt: Backtest) -> pd.DataFrame:
    """MASE e WAPE de cada modelo em cada horizonte, com intervalo de 95% por bootstrap em blocos."""
    real = bt.real[bt.avaliadas]
    linhas = []
    for modelo in MODELOS:
        erros = _erros(bt, modelo)
        for h in bt.horizontes:
            por_origem = metricas.mase_por_origem(erros[:, h - 1], bt.escalas)
            soma_erro, soma_real = erros[:, h - 1].sum(axis=1), real[:, h - 1].sum(axis=1)
            mase_inf, mase_sup = metricas.intervalo_bootstrap(por_origem, bt.indices)
            wape_inf, wape_sup = metricas.intervalo_bootstrap(soma_erro, bt.indices, soma_real)
            linhas.append(
                {
                    "modelo": modelo,
                    "horizonte": int(h),
                    "mase": float(por_origem.mean()),
                    "mase_inf": mase_inf,
                    "mase_sup": mase_sup,
                    "wape": float(soma_erro.sum() / soma_real.sum()),
                    "wape_inf": wape_inf,
                    "wape_sup": wape_sup,
                }
            )
    return pd.DataFrame(linhas).set_index(["modelo", "horizonte"])


def _mase_medio_por_origem(bt: Backtest, modelo: str) -> np.ndarray:
    """Contribuição de cada origem para o MASE médio dos oito horizontes."""
    return np.mean(_erros(bt, modelo) / bt.escalas, axis=(1, 2))


def comparacao_com_a_linha_de_base(bt: Backtest, base: str = "sazonal") -> pd.DataFrame:
    """Cada modelo contra a sazonal ingênua, na média dos horizontes.

    A razão é o MASE do modelo dividido pelo da linha de base, com as mesmas origens sorteadas
    para os dois. Intervalo inteiro abaixo de 1: o modelo erra menos. Intervalo que contém 1: empate.
    """
    real = bt.real[bt.avaliadas]
    referencia = _mase_medio_por_origem(bt, base)
    linhas = []
    for modelo, nome in MODELOS.items():
        erros = _erros(bt, modelo)
        por_origem = _mase_medio_por_origem(bt, modelo)
        soma_erro, soma_real = erros.sum(axis=(1, 2)), real.sum(axis=(1, 2))
        mase_inf, mase_sup = metricas.intervalo_bootstrap(por_origem, bt.indices)
        wape_inf, wape_sup = metricas.intervalo_bootstrap(soma_erro, bt.indices, soma_real)
        razao_inf, razao_sup = metricas.intervalo_bootstrap(por_origem, bt.indices, referencia)
        razao = float(por_origem.sum() / referencia.sum())
        if modelo == base:
            veredito = "linha de base"
        elif razao_sup < 1:
            veredito = "erra menos"
        elif razao_inf > 1:
            veredito = "erra mais"
        else:
            veredito = "empate"
        linhas.append(
            {
                "modelo": modelo,
                "nome": nome,
                "mase": float(por_origem.mean()),
                "mase_inf": mase_inf,
                "mase_sup": mase_sup,
                "wape": float(soma_erro.sum() / soma_real.sum()),
                "wape_inf": wape_inf,
                "wape_sup": wape_sup,
                "razao_para_a_sazonal": razao,
                "razao_inf": razao_inf,
                "razao_sup": razao_sup,
                "veredito": veredito,
            }
        )
    return pd.DataFrame(linhas).set_index("modelo")


def sensibilidade_a_folha(bt: Backtest, folha_inicial: int = 20) -> dict:
    """O gradient boosting com o tamanho mínimo de folha do primeiro desenho (20), para comparação.

    O valor usado no estudo (5) foi escolhido depois de um teste sintético e das origens de
    aquecimento. Este é o resultado que o desenho inicial teria dado na janela de avaliação:
    serve para mostrar que a conclusão não depende dessa escolha.
    """
    previsao = previsores.gradient_boosting(
        bt.y, bt.semanas, bt.origens, HORIZONTE_MAX, parametros={"min_data_in_leaf": folha_inicial}
    )
    erros = np.abs(previsao - bt.real)[bt.avaliadas]
    por_origem = np.mean(erros / bt.escalas, axis=(1, 2))
    referencia = _mase_medio_por_origem(bt, "sazonal")
    inferior, superior = metricas.intervalo_bootstrap(por_origem, bt.indices, referencia)
    return {
        "folha_do_estudo": previsores.PARAMETROS_GBM["min_data_in_leaf"],
        "folha_inicial": folha_inicial,
        "mase_com_a_folha_inicial": float(por_origem.mean()),
        "razao_para_a_sazonal_com_a_folha_inicial": float(por_origem.sum() / referencia.sum()),
        "razao_inf": inferior,
        "razao_sup": superior,
    }


def razao_por_horizonte(bt: Backtest, modelo: str, base: str = "sazonal") -> pd.DataFrame:
    """MASE do modelo dividido pelo da linha de base em cada horizonte, com intervalo pareado."""
    linhas = []
    for h in bt.horizontes:
        numerador = metricas.mase_por_origem(_erros(bt, modelo)[:, h - 1], bt.escalas)
        denominador = metricas.mase_por_origem(_erros(bt, base)[:, h - 1], bt.escalas)
        inferior, superior = metricas.intervalo_bootstrap(numerador, bt.indices, denominador)
        linhas.append(
            {"horizonte": int(h), "razao": float(numerador.sum() / denominador.sum()), "razao_inf": inferior, "razao_sup": superior}
        )
    return pd.DataFrame(linhas).set_index("horizonte")


def erro_por_artigo(bt: Backtest) -> pd.DataFrame:
    """MASE de cada modelo por artigo (média dos horizontes) e a razão do gradient boosting para a sazonal."""
    linhas = []
    for s, artigo in enumerate(bt.artigos):
        linha: dict = {"artigo": dados.rotulo(artigo), "tem_data_calculavel": bool(dados.ARTIGOS[artigo]["eventos"])}
        for modelo in MODELOS:
            linha[f"mase_{modelo}"] = float(_erros(bt, modelo)[:, :, s].mean() / bt.escalas[s])
        for modelo in ("gbm", "ets"):
            numerador = _erros(bt, modelo)[:, :, s].mean(axis=1)
            denominador = _erros(bt, "sazonal")[:, :, s].mean(axis=1)
            inferior, superior = metricas.intervalo_bootstrap(numerador, bt.indices, denominador)
            linha[f"razao_{modelo}"] = float(numerador.sum() / denominador.sum())
            linha[f"razao_{modelo}_inf"], linha[f"razao_{modelo}_sup"] = inferior, superior
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("artigo")


# ---------------------------------------------------------------- intervalos de previsão


def pontuacoes(bt: Backtest, modelo: str) -> np.ndarray:
    """Erro absoluto na escala log(1 + y): a medida de não conformidade dos intervalos.

    Na escala log o intervalo é multiplicativo na escala original ("de metade ao dobro"), o que
    combina com séries que vão de centenas a centenas de milhares de visualizações.
    """
    return np.abs(np.log1p(bt.real) - np.log1p(bt.previsoes[modelo]))


def raios_dos_intervalos(
    bt: Backtest, modelo: str = "ets", gama: float = GAMA, metodos: tuple[str, ...] = tuple(METODOS)
) -> dict[tuple[str, float], np.ndarray]:
    """Meia-largura (na escala log) do intervalo de cada método, em cada origem, horizonte e série.

    - paramétrico: z · desvio declarado pelo modelo (só para a suavização exponencial);
    - conformal simples: quantil dos erros fora da amostra já conhecidos na origem;
    - conformal adaptativo: o mesmo quantil com o nível corrigido a cada erro (ACI);
    - conformal por grupo: semanas de data comemorativa calibradas à parte, juntando os artigos
      (cada artigo tem poucas por ano); as demais semanas, calibradas por artigo.
    Os três primeiros calibram por artigo. Com horizonte h, o erro da origem t só é conhecido na
    origem t + h: esse é o atraso.
    """
    pont = pontuacoes(bt, modelo)
    n_series = pont.shape[2]
    de_evento = semanas_de_evento(bt) == 1
    raios: dict[tuple[str, float], np.ndarray] = {}
    for nivel in NIVEIS:
        alfa = 1 - nivel
        if modelo == "ets" and "parametrico" in metodos:
            raios[("parametrico", nivel)] = stats.norm.ppf(1 - alfa / 2) * bt.desvio_ets
        for metodo in ("conformal", "aci", "grupo"):
            if metodo not in metodos:
                continue
            raio = np.empty_like(pont)
            for h in bt.horizontes:
                if metodo == "grupo":
                    # Rótulo de grupo: o número do artigo nas semanas comuns; um rótulo só para toda semana de evento.
                    grupos = np.where(de_evento[:, h - 1], n_series, np.arange(n_series)[None, :])
                    raio[:, h - 1] = conformal.conformal_por_grupo(
                        pont[:, h - 1], grupos, alfa, atraso=int(h), minimo=MINIMO_CALIBRACAO
                    )
                    continue
                for s in range(n_series):
                    serie = pont[:, h - 1, s]
                    if metodo == "conformal":
                        raio[:, h - 1, s] = conformal.conformal_simples(serie, alfa, atraso=int(h), minimo=MINIMO_CALIBRACAO)
                    else:
                        raio[:, h - 1, s], _ = conformal.conformal_adaptativo(
                            serie, alfa, gama=gama, atraso=int(h), minimo=MINIMO_CALIBRACAO
                        )
            raios[(metodo, nivel)] = raio
    for chave, raio in raios.items():
        if np.isnan(raio[bt.avaliadas]).any():
            raise ValueError(f"Intervalo sem calibração suficiente na janela de avaliação: {chave}.")
    return raios


def _fator_mediano(raio: np.ndarray) -> float:
    """Largura típica do intervalo como fator multiplicativo: a faixa vai de previsão ÷ fator a previsão × fator."""
    return float(np.exp(np.median(np.maximum(raio, 0.0))))


def cobertura_dos_intervalos(bt: Backtest, pont: np.ndarray, raios: dict[tuple[str, float], np.ndarray]) -> pd.DataFrame:
    """Cobertura real de cada método e nível, por horizonte e no geral (horizonte 0 = todos).

    Cobertura: fração das previsões avaliadas em que o valor real ficou dentro do intervalo.
    """
    pont = pont[bt.avaliadas]
    linhas = []
    for (metodo, nivel), raio in raios.items():
        raio = raio[bt.avaliadas]
        cobre = pont <= raio
        for h in [0, *bt.horizontes]:
            recorte = (slice(None), slice(None)) if h == 0 else (slice(None), h - 1)
            por_origem = cobre[recorte].reshape(len(cobre), -1).mean(axis=1)
            inferior, superior = metricas.intervalo_bootstrap(por_origem, bt.indices)
            linhas.append(
                {
                    "metodo": metodo,
                    "nivel": nivel,
                    "horizonte": int(h),
                    "cobertura": float(por_origem.mean()),
                    "cobertura_inf": inferior,
                    "cobertura_sup": superior,
                    "fator_mediano": _fator_mediano(raio[recorte]),
                    "intervalos_infinitos": float(np.mean(np.isinf(raio[recorte]))),
                }
            )
    return pd.DataFrame(linhas).set_index(["metodo", "nivel", "horizonte"])


def cobertura_por_ano(bt: Backtest, pont: np.ndarray, raios: dict[tuple[str, float], np.ndarray], nivel: float = 0.80) -> pd.DataFrame:
    """Cobertura real em cada ano da janela de avaliação (ano da semana em que a previsão foi feita)."""
    anos = bt.semanas[bt.origens[bt.avaliadas]].year
    tabela = {}
    for metodo in METODOS:
        cobre = (pont <= raios[(metodo, nivel)])[bt.avaliadas]
        tabela[metodo] = pd.Series(cobre.reshape(len(cobre), -1).mean(axis=1)).groupby(anos.to_numpy()).mean()
    resultado = pd.DataFrame(tabela)
    resultado.index.name = "ano"
    resultado["origens"] = pd.Series(anos.to_numpy()).value_counts().sort_index()
    return resultado


def semanas_de_evento(bt: Backtest) -> np.ndarray:
    """Máscara (origens, horizontes, séries): a semana prevista é a da data comemorativa do artigo ou a anterior.

    Artigos sem data calculável (Imposto de renda e Enem) ficam com NaN: não entram nesse recorte.
    """
    marcada = np.full(bt.y.shape, np.nan)
    for s, artigo in enumerate(bt.artigos):
        eventos = dados.ARTIGOS[artigo]["eventos"]
        if not eventos:
            continue
        perto = np.zeros(len(bt.semanas), dtype=bool)
        for evento in eventos:
            perto |= np.isin(calendario.semanas_ate_evento(bt.semanas, evento), SEMANAS_DE_EVENTO)
        marcada[:, s] = perto
    return validacao.realizado(marcada, bt.origens, HORIZONTE_MAX)


def cobertura_em_eventos(bt: Backtest, pont: np.ndarray, raios: dict[tuple[str, float], np.ndarray]) -> pd.DataFrame:
    """Cobertura nas semanas de data comemorativa contra as demais semanas.

    A garantia dos métodos conformes é na média de todas as semanas. Para quem planeja verba, a
    semana que importa é a do evento; esta tabela mede a cobertura justamente ali.
    """
    marcada = semanas_de_evento(bt)[bt.avaliadas]
    pont = pont[bt.avaliadas]
    recortes = {"semana de evento": marcada == 1, "demais semanas": marcada == 0}
    linhas = []
    for (metodo, nivel), raio in raios.items():
        raio = raio[bt.avaliadas]
        cobre = pont <= raio
        for nome, recorte in recortes.items():
            for h in [0, *bt.horizontes]:
                fatia = slice(None) if h == 0 else slice(h - 1, h)
                dentro = recorte[:, fatia]
                numerador = (cobre[:, fatia] & dentro).sum(axis=(1, 2)).astype(float)
                denominador = dentro.sum(axis=(1, 2)).astype(float)
                inferior, superior = metricas.intervalo_bootstrap(numerador, bt.indices, denominador)
                linhas.append(
                    {
                        "metodo": metodo,
                        "nivel": nivel,
                        "recorte": nome,
                        "horizonte": int(h),
                        "previsoes": int(denominador.sum()),
                        "cobertura": float(numerador.sum() / denominador.sum()),
                        "cobertura_inf": inferior,
                        "cobertura_sup": superior,
                        "fator_mediano": _fator_mediano(raio[:, fatia][dentro]),
                    }
                )
    return pd.DataFrame(linhas).set_index(["metodo", "nivel", "recorte", "horizonte"])


def conformal_por_modelo(bt: Backtest, nivel: float = 0.80) -> pd.DataFrame:
    """Intervalos conformes em volta de cada previsor: a cobertura se iguala, a largura não.

    O método conformal aceita qualquer previsão pontual. A cobertura geral fica perto do
    prometido em todos; o que muda é a largura, que é o preço de prever pior.
    """
    linhas = []
    for modelo, nome in MODELOS.items():
        pont = pontuacoes(bt, modelo)
        todos = raios_dos_intervalos(bt, modelo, metodos=("aci", "grupo"))
        raios = {chave: raio for chave, raio in todos.items() if chave[1] == nivel}
        geral = cobertura_dos_intervalos(bt, pont, raios).xs(0, level="horizonte")
        eventos = cobertura_em_eventos(bt, pont, raios).xs(0, level="horizonte")
        for metodo in ("aci", "grupo"):
            evento, demais = eventos.loc[(metodo, nivel, "semana de evento")], eventos.loc[(metodo, nivel, "demais semanas")]
            linhas.append(
                {
                    "modelo": modelo,
                    "nome": nome,
                    "metodo": metodo,
                    **geral.loc[(metodo, nivel)][["cobertura", "cobertura_inf", "cobertura_sup", "fator_mediano"]].to_dict(),
                    "cobertura_evento": evento["cobertura"],
                    "cobertura_evento_inf": evento["cobertura_inf"],
                    "cobertura_evento_sup": evento["cobertura_sup"],
                    "fator_evento": evento["fator_mediano"],
                    "fator_demais": demais["fator_mediano"],
                }
            )
    return pd.DataFrame(linhas).set_index(["modelo", "metodo"])


def sensibilidade_ao_gama(bt: Backtest, modelo: str = "ets") -> pd.DataFrame:
    """Cobertura e largura da ACI para outros passos de correção (o estudo usa um só, fixado antes)."""
    pont = pontuacoes(bt, modelo)
    linhas = []
    for gama in GAMAS_TESTADOS:
        so_aci = raios_dos_intervalos(bt, modelo, gama=gama, metodos=("aci",))
        geral = cobertura_dos_intervalos(bt, pont, so_aci).xs(0, level="horizonte").reset_index()
        for _, linha in geral.iterrows():
            linhas.append({"gama": gama, **linha.drop("metodo").to_dict()})
    return pd.DataFrame(linhas).set_index(["gama", "nivel"])


def serie_de_exemplo(
    bt: Backtest, raios: dict[tuple[str, float], np.ndarray], artigo: str, h: int, modelo: str = "ets", nivel: float = 0.80
) -> pd.DataFrame:
    """Previsão a h semanas de um artigo, na janela de avaliação, com as faixas simples e por grupo."""
    s = bt.artigos.index(artigo)
    alvo = bt.origens[bt.avaliadas] + h - 1
    centro = np.log1p(bt.previsoes[modelo][bt.avaliadas, h - 1, s])
    tabela = pd.DataFrame(
        {"real": bt.real[bt.avaliadas, h - 1, s], "previsto": bt.previsoes[modelo][bt.avaliadas, h - 1, s]},
        index=bt.semanas[alvo],
    )
    for metodo in ("conformal", "grupo"):
        raio = raios[(metodo, nivel)][bt.avaliadas, h - 1, s]
        tabela[f"{metodo}_inf"] = np.expm1(centro - np.maximum(raio, 0.0))
        tabela[f"{metodo}_sup"] = np.expm1(centro + np.maximum(raio, 0.0))
    tabela.index.name = "semana"
    return tabela


# ---------------------------------------------------------------- tudo junto


@dataclass
class Resultados:
    """As tabelas do estudo, na ordem em que o README as usa."""

    backtest: Backtest
    base: dict
    comparacao: pd.DataFrame
    erro_por_horizonte: pd.DataFrame
    razao_por_horizonte: dict[str, pd.DataFrame]
    erro_por_artigo: pd.DataFrame
    raios: dict[tuple[str, float], np.ndarray]
    cobertura: pd.DataFrame
    cobertura_por_ano: pd.DataFrame
    cobertura_em_eventos: pd.DataFrame
    conformal_por_modelo: pd.DataFrame
    sensibilidade_ao_gama: pd.DataFrame
    sensibilidade_a_folha: dict
    exemplo: pd.DataFrame
    achados: dict


ARTIGO_DE_EXEMPLO, HORIZONTE_DE_EXEMPLO = "Black_Friday", 4


def achados(r: "Resultados") -> dict:
    """Os números que sustentam o texto do README, reunidos num lugar só."""
    comparacao, cobertura, eventos = r.comparacao, r.cobertura, r.cobertura_em_eventos
    por_artigo = r.erro_por_artigo
    saida: dict = {
        "reducao_do_erro_gbm": 1 - comparacao.loc["gbm", "razao_para_a_sazonal"],
        "reducao_do_erro_gbm_inf": 1 - comparacao.loc["gbm", "razao_sup"],
        "reducao_do_erro_gbm_sup": 1 - comparacao.loc["gbm", "razao_inf"],
        "reducao_do_erro_ets": 1 - comparacao.loc["ets", "razao_para_a_sazonal"],
        "reducao_do_erro_ets_inf": 1 - comparacao.loc["ets", "razao_sup"],
        "reducao_do_erro_ets_sup": 1 - comparacao.loc["ets", "razao_inf"],
        "artigos_em_que_o_gbm_erra_menos": int((por_artigo["razao_gbm_sup"] < 1).sum()),
        "artigos_em_empate_gbm": int(((por_artigo["razao_gbm_inf"] <= 1) & (por_artigo["razao_gbm_sup"] >= 1)).sum()),
        "artigos_em_que_o_gbm_erra_mais": int((por_artigo["razao_gbm_inf"] > 1).sum()),
        "horizontes_em_que_o_ets_empata": [
            int(h) for h, linha in r.razao_por_horizonte["ets"].iterrows() if linha["razao_inf"] <= 1 <= linha["razao_sup"]
        ],
    }
    for nivel in NIVEIS:
        sufixo = f"{round(nivel * 100)}"
        for metodo in METODOS:
            geral = cobertura.loc[(metodo, nivel, 0)]
            evento = eventos.loc[(metodo, nivel, "semana de evento", 0)]
            demais = eventos.loc[(metodo, nivel, "demais semanas", 0)]
            saida[f"cobertura_{sufixo}_{metodo}"] = geral["cobertura"]
            saida[f"cobertura_{sufixo}_{metodo}_evento"] = evento["cobertura"]
            saida[f"cobertura_{sufixo}_{metodo}_demais"] = demais["cobertura"]
            saida[f"fator_{sufixo}_{metodo}"] = geral["fator_mediano"]
            saida[f"fator_{sufixo}_{metodo}_evento"] = evento["fator_mediano"]
            saida[f"fator_{sufixo}_{metodo}_demais"] = demais["fator_mediano"]
            saida[f"infinitos_{sufixo}_{metodo}"] = geral["intervalos_infinitos"]
    saida["previsoes_em_semana_de_evento"] = int(eventos.loc[("conformal", 0.80, "semana de evento", 0), "previsoes"])
    saida["previsoes_nas_demais_semanas"] = int(eventos.loc[("conformal", 0.80, "demais semanas", 0), "previsoes"])
    return {chave: (float(valor) if isinstance(valor, (np.floating, float)) else valor) for chave, valor in saida.items()}


def rodar(semanal: pd.DataFrame) -> Resultados:
    """A análise inteira, do backtest às tabelas. É o que o script e o notebook chamam."""
    bt = rodar_backtest(semanal)
    pont = pontuacoes(bt, "ets")
    raios = raios_dos_intervalos(bt, "ets")
    r = Resultados(
        backtest=bt,
        base=resumo_da_base(bt),
        comparacao=comparacao_com_a_linha_de_base(bt),
        erro_por_horizonte=erro_por_horizonte(bt),
        razao_por_horizonte={modelo: razao_por_horizonte(bt, modelo) for modelo in ("gbm", "ets")},
        erro_por_artigo=erro_por_artigo(bt),
        raios=raios,
        cobertura=cobertura_dos_intervalos(bt, pont, raios),
        cobertura_por_ano=cobertura_por_ano(bt, pont, raios),
        cobertura_em_eventos=cobertura_em_eventos(bt, pont, raios),
        conformal_por_modelo=conformal_por_modelo(bt),
        sensibilidade_ao_gama=sensibilidade_ao_gama(bt),
        sensibilidade_a_folha=sensibilidade_a_folha(bt),
        exemplo=serie_de_exemplo(bt, raios, ARTIGO_DE_EXEMPLO, HORIZONTE_DE_EXEMPLO),
        achados={},
    )
    r.achados = achados(r)
    return r
