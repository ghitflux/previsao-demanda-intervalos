"""Bloco `para_o_site` de reports/resultados.json: o resumo trilíngue que o site do portfólio desenha.

Todo número dos textos vem das tabelas da análise; nada é digitado à mão.
"""

from __future__ import annotations

from previsao_demanda.analise import Resultados

METODOS_NO_GRAFICO = ["parametrico", "conformal", "aci", "grupo"]
ROTULOS = {
    "pt": ["Paramétrico do modelo", "Conformal simples", "Conformal adaptativo", "Conformal por grupo"],
    "en": ["Model's parametric", "Split conformal", "Adaptive conformal", "Group conformal"],
    "es": ["Paramétrico del modelo", "Conformal simple", "Conformal adaptativo", "Conformal por grupo"],
}


def _pct(valor: float, idioma: str) -> str:
    """Percentual inteiro no formato do idioma (em espanhol, com espaço antes do símbolo)."""
    return f"{round(valor * 100)} %" if idioma == "es" else f"{round(valor * 100)}%"


def _num(valor: float, idioma: str, casas: int = 1) -> str:
    texto = f"{valor:.{casas}f}"
    return texto if idioma == "en" else texto.replace(".", ",")


def para_o_site(r: Resultados) -> dict:
    a, base, eventos = r.achados, r.base, r.cobertura_em_eventos
    nivel = 0.80
    coberturas = [a[f"cobertura_80_{m}"] for m in ("parametrico", "conformal", "aci")]
    minimo, maximo = min(coberturas), max(coberturas)
    evento = a["cobertura_80_parametrico_evento"]
    grupo = a["cobertura_80_grupo_evento"]
    fator_simples, fator_grupo = a["fator_80_conformal_evento"], a["fator_80_grupo_evento"]
    reducao, red_inf, red_sup = a["reducao_do_erro_gbm"], a["reducao_do_erro_gbm_inf"], a["reducao_do_erro_gbm_sup"]
    milhoes = base["visualizacoes_no_periodo"] / 1e6

    def p(valor: float) -> dict:
        return {idioma: _pct(valor, idioma) for idioma in ("pt", "en", "es")}

    def n(valor: float, casas: int = 1) -> dict:
        return {idioma: _num(valor, idioma, casas) for idioma in ("pt", "en", "es")}

    series = []
    nomes = {
        "semana de evento": {"pt": "Semana da data comemorativa", "en": "Holiday week", "es": "Semana de la fecha señalada"},
        "demais semanas": {"pt": "Demais semanas", "en": "All other weeks", "es": "Resto de las semanas"},
    }
    for recorte, papel in (("semana de evento", "principal"), ("demais semanas", "neutro")):
        series.append(
            {
                "nome": nomes[recorte],
                "papel": papel,
                "pontos": [
                    [rotulo, round(float(eventos.loc[(metodo, nivel, recorte, 0), "cobertura"]), 4)]
                    for rotulo, metodo in zip(ROTULOS["pt"], METODOS_NO_GRAFICO)
                ],
                "rotulos": ROTULOS,
            }
        )

    return {
        "numero_chave": {
            "pt": f"{_pct(nivel, 'pt')} → {_pct(evento, 'pt')}",
            "en": f"{_pct(nivel, 'en')} → {_pct(evento, 'en')}",
            "es": f"{_pct(nivel, 'es')} → {_pct(evento, 'es')}",
        },
        "pergunta": {
            "pt": "Quanto interesse vai haver nas próximas semanas, e quanto dá para confiar na previsão?",
            "en": "How much interest will there be in the coming weeks, and how far can the forecast be trusted?",
            "es": "¿Cuánto interés habrá en las próximas semanas y hasta dónde se puede confiar en la previsión?",
        },
        "resultado": {
            "pt": (
                f"Na média de todas as semanas, o intervalo de {p(nivel)['pt']} entrega quase o que promete: cobre de "
                f"{p(minimo)['pt']} a {p(maximo)['pt']} dos casos, conforme o método. Na semana da data comemorativa, a que "
                f"decide a verba, ele cobre só {p(evento)['pt']}; calibrando essas semanas à parte, a cobertura volta a "
                f"{p(grupo)['pt']}, com uma faixa que vai da previsão dividida por {n(fator_grupo)['pt']} à previsão "
                f"multiplicada por {n(fator_grupo)['pt']}."
            ),
            "en": (
                f"Averaged over every week, the {p(nivel)['en']} interval nearly keeps its promise: it covers "
                f"{p(minimo)['en']} to {p(maximo)['en']} of outcomes, depending on the method. In the holiday week itself, "
                f"the one that drives the budget, it covers only {p(evento)['en']}; calibrating those weeks separately "
                f"brings coverage back to {p(grupo)['en']}, with a band that runs from the forecast divided by "
                f"{n(fator_grupo)['en']} to the forecast multiplied by {n(fator_grupo)['en']}."
            ),
            "es": (
                f"En el promedio de todas las semanas, el intervalo del {p(nivel)['es']} casi cumple lo que promete: cubre "
                f"entre el {p(minimo)['es']} y el {p(maximo)['es']} de los casos, según el método. En la semana de la fecha "
                f"señalada, la que decide el presupuesto, cubre solo el {p(evento)['es']}; al calibrar esas semanas por "
                f"separado, la cobertura vuelve al {p(grupo)['es']}, con una banda que va de la previsión dividida entre "
                f"{n(fator_grupo)['es']} a la previsión multiplicada por {n(fator_grupo)['es']}."
            ),
        },
        "detalhe": {
            "pt": (
                f"No erro da previsão em si, o gradient boosting com variáveis de calendário erra {p(reducao)['pt']} menos "
                f"do que repetir a mesma semana do ano anterior (intervalo de 95%: {p(red_inf)['pt']} a {p(red_sup)['pt']}). "
                f"No gráfico, cada par de barras é um jeito de construir o intervalo de {p(nivel)['pt']}: os três primeiros "
                f"usam praticamente a mesma largura o ano inteiro (a previsão dividida ou multiplicada por cerca de "
                f"{n(fator_simples)['pt']}) e falham justamente no pico."
            ),
            "en": (
                f"On the point forecast, gradient boosting with calendar features cuts the error by {p(reducao)['en']} "
                f"compared with simply repeating the same week of the previous year (95% interval: {p(red_inf)['en']} to "
                f"{p(red_sup)['en']}). In the chart, each pair of bars is one way of building the {p(nivel)['en']} interval: "
                f"the first three keep practically the same width all year (the forecast divided or multiplied by about "
                f"{n(fator_simples)['en']}) and fail exactly at the peak."
            ),
            "es": (
                f"En la previsión puntual, el gradient boosting con variables de calendario comete un {p(reducao)['es']} "
                f"menos de error que repetir la misma semana del año anterior (intervalo del 95 %: {p(red_inf)['es']} a "
                f"{p(red_sup)['es']}). En el gráfico, cada par de barras es una forma de construir el intervalo del "
                f"{p(nivel)['es']}: las tres primeras usan prácticamente el mismo ancho todo el año (la previsión dividida "
                f"o multiplicada por cerca de {n(fator_simples)['es']}) y fallan justo en el pico."
            ),
        },
        "dados": {
            "pt": (
                f"Visualizações semanais de {base['artigos']} artigos da Wikipédia em português, de julho de 2015 a outubro "
                f"de 2026 ({base['semanas']} semanas, {n(milhoes)['pt']} milhões de visualizações): um indicador de "
                f"interesse, não de venda."
            ),
            "en": (
                f"Weekly page views of {base['artigos']} Portuguese-language Wikipedia articles, July 2015 to October 2026 "
                f"({base['semanas']} weeks, {n(milhoes)['en']} million views): a proxy for interest, not for sales."
            ),
            "es": (
                f"Visitas semanales de {base['artigos']} artículos de Wikipedia en portugués, de julio de 2015 a octubre de "
                f"2026 ({base['semanas']} semanas, {n(milhoes)['es']} millones de visitas): un indicador de interés, no de "
                f"ventas."
            ),
        },
        "etiquetas": ["séries temporais", "predição conforme", "backtesting", "LightGBM", "quantificação de incerteza"],
        "grafico": {
            "tipo": "barras",
            "titulo": {
                "pt": f"Cobertura real do intervalo de {p(nivel)['pt']}, na semana da data comemorativa e nas demais",
                "en": f"Actual coverage of the {p(nivel)['en']} interval, in holiday weeks and in all other weeks",
                "es": f"Cobertura real del intervalo del {p(nivel)['es']}, en la semana de la fecha señalada y en el resto",
            },
            "eixo_x": {"pt": "Como o intervalo é construído", "en": "How the interval is built", "es": "Cómo se construye el intervalo"},
            "eixo_y": {
                "pt": f"Cobertura real (prometido: {p(nivel)['pt']})",
                "en": f"Actual coverage (promised: {p(nivel)['en']})",
                "es": f"Cobertura real (prometido: {p(nivel)['es']})",
            },
            "formato_x": "numero",
            "formato_y": "percentual",
            "rotulos": ROTULOS,
            "series": series,
        },
    }
