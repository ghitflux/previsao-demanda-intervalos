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
        "graficos_extras": graficos_extras(r),
    }


# ---------------------------------------------------------------- gráficos extras (as outras figuras do README)

IDIOMAS = ("pt", "en", "es")
NOMES_MODELOS = {
    "gbm": {"pt": "Gradient boosting com calendário", "en": "Gradient boosting with calendar", "es": "Gradient boosting con calendario"},
    "sazonal": {"pt": "Sazonal ingênua", "en": "Seasonal naive", "es": "Estacional ingenua"},
    "ets": {"pt": "Suavização exponencial", "en": "Exponential smoothing", "es": "Suavizado exponencial"},
    "media_movel": {"pt": "Média móvel de 4 semanas", "en": "4-week moving average", "es": "Media móvil de 4 semanas"},
}
PAPEL_MODELOS = {"gbm": "principal", "sazonal": "base", "ets": "alternativa", "media_movel": "neutro"}
NOMES_METODOS = {
    "aci": {"pt": "Conformal adaptativo (ACI)", "en": "Adaptive conformal (ACI)", "es": "Conformal adaptativo (ACI)"},
    "parametrico": {"pt": "Paramétrico do modelo", "en": "Model's parametric", "es": "Paramétrico del modelo"},
    "conformal": {"pt": "Conformal simples", "en": "Split conformal", "es": "Conformal simple"},
}
PAPEL_METODOS = {"aci": "principal", "parametrico": "base", "conformal": "alternativa"}
ARTIGOS_TRADUZIDOS = {
    "Black Friday": {"en": "Black Friday", "es": "Black Friday"},
    "Dia das Mães": {"en": "Mother's Day", "es": "Día de la Madre"},
    "Dia dos Pais": {"en": "Father's Day", "es": "Día del Padre"},
    "Dia dos Namorados": {"en": "Valentine's Day", "es": "Día de los Enamorados"},
    "Natal": {"en": "Christmas", "es": "Navidad"},
    "Páscoa": {"en": "Easter", "es": "Pascua"},
    "Carnaval": {"en": "Carnival", "es": "Carnaval"},
    "Festa junina": {"en": "June festivals", "es": "Fiestas de junio"},
    "Imposto de renda": {"en": "Income tax", "es": "Impuesto sobre la renta"},
    "Enem": {"en": "Enem (national exam)", "es": "Enem (examen nacional)"},
}
SEMANAS_NO_EXEMPLO = 80  # limite de pontos por série: as últimas 80 semanas, que incluem a Black Friday de 2025


def _txt(pt: str, en: str, es: str) -> dict:
    return {"pt": pt, "en": en, "es": es}


def _grafico_erro_por_horizonte(r: Resultados) -> dict:
    erros, a = r.erro_por_horizonte, r.achados
    p = {i: _pct(a["reducao_do_erro_gbm"], i) for i in IDIOMAS}
    gbm1, saz1 = erros.loc[("gbm", 1), "mase"], erros.loc[("sazonal", 1), "mase"]
    series = []
    for modelo in ("gbm", "sazonal", "ets", "media_movel"):
        tabela = erros.loc[modelo]
        series.append(
            {
                "nome": NOMES_MODELOS[modelo],
                "papel": PAPEL_MODELOS[modelo],
                "pontos": [[int(h), round(float(v), 4)] for h, v in tabela["mase"].items()],
            }
        )
    tabela_gbm = erros.loc["gbm"]
    return {
        "tipo": "linhas",
        "titulo": _txt(
            f"O gradient boosting com calendário erra {p['pt']} menos que repetir o ano anterior",
            f"Gradient boosting with calendar features errs {p['en']} less than repeating last year",
            f"El gradient boosting con calendario se equivoca un {p['es']} menos que repetir el año anterior",
        ),
        "eixo_x": _txt("Horizonte (semanas à frente)", "Horizon (weeks ahead)", "Horizonte (semanas por delante)"),
        "eixo_y": _txt("MASE (menor é melhor)", "MASE (lower is better)", "MASE (menor es mejor)"),
        "formato_x": "numero",
        "formato_y": "numero",
        "series": series,
        "faixa": {
            "serie": 0,
            "inferior": [[int(h), round(float(v), 4)] for h, v in tabela_gbm["mase_inf"].items()],
            "superior": [[int(h), round(float(v), 4)] for h, v in tabela_gbm["mase_sup"].items()],
        },
        "leitura": _txt(
            f"Erro de cada previsor de 1 a 8 semanas à frente, relativo ao erro histórico da sazonal ingênua. O modelo "
            f"fica abaixo da linha de base em todos os horizontes (a 1 semana, {_num(gbm1, 'pt', 2)} contra "
            f"{_num(saz1, 'pt', 2)}); a média móvel chega atrasada em todo pico.",
            f"Error of each forecaster 1 to 8 weeks ahead, relative to the seasonal naive's historical error. The model "
            f"stays below the baseline at every horizon (one week out, {_num(gbm1, 'en', 2)} versus "
            f"{_num(saz1, 'en', 2)}); the moving average lags behind every peak.",
            f"Error de cada previsor de 1 a 8 semanas vista, relativo al error histórico de la estacional ingenua. El "
            f"modelo queda por debajo de la referencia en todos los horizontes (a una semana, {_num(gbm1, 'es', 2)} "
            f"frente a {_num(saz1, 'es', 2)}); la media móvil llega tarde a cada pico.",
        ),
    }


def _grafico_cobertura_95(r: Resultados) -> dict:
    cobertura = r.cobertura.sort_index()
    nivel = 0.95
    entregue = cobertura.loc[("parametrico", nivel, 0), "cobertura"]
    conforme = cobertura.loc[("conformal", nivel, 0), "cobertura"]
    series = []
    for metodo in ("aci", "parametrico", "conformal"):
        tabela = cobertura.loc[(metodo, nivel)].drop(index=0)
        series.append(
            {
                "nome": NOMES_METODOS[metodo],
                "papel": PAPEL_METODOS[metodo],
                "pontos": [[int(h), round(float(v), 4)] for h, v in tabela["cobertura"].items()],
            }
        )
    param = cobertura.loc[("parametrico", nivel)].drop(index=0)
    pe, ce = {i: _pct(entregue, i) for i in IDIOMAS}, {i: _pct(conforme, i) for i in IDIOMAS}
    return {
        "tipo": "linhas",
        "titulo": _txt(
            f"O intervalo paramétrico promete 95% e entrega {pe['pt']}; os conformes entregam",
            f"The parametric interval promises 95% and delivers {pe['en']}; the conformal ones deliver",
            f"El intervalo paramétrico promete un 95 % y entrega un {pe['es']}; los conformales cumplen",
        ),
        "eixo_x": _txt("Horizonte (semanas à frente)", "Horizon (weeks ahead)", "Horizonte (semanas por delante)"),
        "eixo_y": _txt("Cobertura real (prometido: 95%)", "Actual coverage (promised: 95%)", "Cobertura real (prometido: 95 %)"),
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": series,
        "faixa": {
            "serie": 1,
            "inferior": [[int(h), round(float(v), 4)] for h, v in param["cobertura_inf"].items()],
            "superior": [[int(h), round(float(v), 4)] for h, v in param["cobertura_sup"].items()],
        },
        "leitura": _txt(
            f"Fração das previsões em que o valor real ficou dentro do intervalo de 95%, juntando todas as semanas. O "
            f"paramétrico, que supõe erro normal, fica em {pe['pt']}; o conformal simples entrega {ce['pt']}, perto do "
            f"prometido.",
            f"Share of forecasts whose actual value fell inside the 95% interval, across all weeks. The parametric "
            f"interval, which assumes normal errors, sits at {pe['en']}; split conformal delivers {ce['en']}, close to "
            f"the promise.",
            f"Proporción de previsiones cuyo valor real cayó dentro del intervalo del 95 %, con todas las semanas. El "
            f"paramétrico, que supone errores normales, se queda en un {pe['es']}; el conformal simple entrega un "
            f"{ce['es']}, cerca de lo prometido.",
        ),
    }


def _grafico_exemplo(r: Resultados) -> dict:
    tabela = r.exemplo.iloc[-SEMANAS_NO_EXEMPLO:]
    datas = [f"{d:%Y-%m-%d}" for d in tabela.index]

    def pontos(coluna: str) -> list:
        return [[d, round(float(v), 1)] for d, v in zip(datas, tabela[coluna])]

    return {
        "tipo": "linhas",
        "titulo": _txt(
            "A faixa calibrada por grupo se abre na semana do evento, onde o erro mora",
            "The group-calibrated band widens in the holiday week, where the error lives",
            "La banda calibrada por grupo se abre en la semana de la fecha, donde está el error",
        ),
        "eixo_x": _txt("Semana", "Week", "Semana"),
        "eixo_y": _txt("Visualizações na semana", "Weekly page views", "Visitas en la semana"),
        "formato_x": "data",
        "formato_y": "numero",
        "series": [
            {"nome": _txt("Previsão (4 semanas antes)", "Forecast (4 weeks ahead)", "Previsión (4 semanas antes)"), "papel": "principal", "pontos": pontos("previsto")},
            {"nome": _txt("Realizado", "Actual", "Real"), "papel": "neutro", "pontos": pontos("real")},
        ],
        "faixa": {"serie": 0, "inferior": pontos("grupo_inf"), "superior": pontos("grupo_sup")},
        "leitura": _txt(
            f"Black Friday nas últimas {SEMANAS_NO_EXEMPLO} semanas: previsão da suavização exponencial feita 4 semanas "
            f"antes, com o intervalo de 80% calibrado por grupo. A faixa é estreita no resto do ano e se abre no pico, "
            f"porque basta o pico chegar uma semana antes ou depois para o erro ser de várias vezes o valor.",
            f"Black Friday over the last {SEMANAS_NO_EXEMPLO} weeks: the exponential smoothing forecast made 4 weeks "
            f"ahead, with the group-calibrated 80% interval. The band is narrow most of the year and widens at the peak, "
            f"because a peak arriving one week early or late means an error of several times the value.",
            f"Black Friday en las últimas {SEMANAS_NO_EXEMPLO} semanas: previsión del suavizado exponencial hecha 4 "
            f"semanas antes, con el intervalo del 80 % calibrado por grupo. La banda es estrecha el resto del año y se "
            f"abre en el pico, porque basta con que llegue una semana antes o después para que el error sea de varias "
            f"veces el valor.",
        ),
    }


def _grafico_cobertura_por_ano(r: Resultados) -> dict:
    por_ano = r.cobertura_por_ano
    ultimo = int(por_ano.index[-1])
    aci = {i: _pct(por_ano["aci"].iloc[-1], i) for i in IDIOMAS}
    simples = {i: _pct(por_ano["conformal"].iloc[-1], i) for i in IDIOMAS}
    series = [
        {
            "nome": NOMES_METODOS[metodo],
            "papel": PAPEL_METODOS[metodo],
            "pontos": [[int(ano), round(float(v), 4)] for ano, v in por_ano[metodo].items()],
        }
        for metodo in ("aci", "parametrico", "conformal")
    ]
    return {
        "tipo": "linhas",
        "titulo": _txt(
            f"Quando os erros cresceram, em {ultimo - 1} e {ultimo}, só o adaptativo ficou perto dos 80%",
            f"When errors grew in {ultimo - 1} and {ultimo}, only the adaptive interval stayed near 80%",
            f"Cuando los errores crecieron, en {ultimo - 1} y {ultimo}, solo el adaptativo se mantuvo cerca del 80 %",
        ),
        "eixo_x": _txt("Ano em que a previsão foi feita", "Year the forecast was made", "Año en que se hizo la previsión"),
        "eixo_y": _txt("Cobertura real (prometido: 80%)", "Actual coverage (promised: 80%)", "Cobertura real (prometido: 80 %)"),
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": series,
        "leitura": _txt(
            f"Cobertura do intervalo de 80% em cada ano ({int(por_ano.index[0])} e {ultimo} são parciais). O conformal "
            f"simples supõe que os erros de amanhã se parecem com os de ontem e caiu para {simples['pt']} em {ultimo}; "
            f"o adaptativo corrige o nível a cada erro e ficou em {aci['pt']}.",
            f"Coverage of the 80% interval in each year ({int(por_ano.index[0])} and {ultimo} are partial). Split "
            f"conformal assumes tomorrow's errors look like yesterday's and dropped to {simples['en']} in {ultimo}; the "
            f"adaptive method adjusts its level after every miss and held at {aci['en']}.",
            f"Cobertura del intervalo del 80 % en cada año ({int(por_ano.index[0])} y {ultimo} son parciales). El "
            f"conformal simple supone que los errores de mañana se parecen a los de ayer y cayó al {simples['es']} en "
            f"{ultimo}; el adaptativo corrige el nivel tras cada fallo y se quedó en el {aci['es']}.",
        ),
    }


def _grafico_erro_por_artigo(r: Resultados) -> dict:
    tabela = r.erro_por_artigo.sort_values("razao_gbm")
    a = r.achados
    rotulos = {"pt": list(tabela.index)}
    for idioma in ("en", "es"):
        rotulos[idioma] = [ARTIGOS_TRADUZIDOS[artigo][idioma] for artigo in tabela.index]
    vence, total = a["artigos_em_que_o_gbm_erra_menos"], len(tabela)
    pior = tabela.index[-1]
    return {
        "tipo": "barras",
        "titulo": _txt(
            f"O modelo erra menos que a sazonal ingênua em {vence} dos {total} artigos",
            f"The model beats the seasonal naive in {vence} of the {total} articles",
            f"El modelo se equivoca menos que la estacional ingenua en {vence} de los {total} artículos",
        ),
        "eixo_x": _txt("Artigo", "Article", "Artículo"),
        "eixo_y": _txt(
            "Erro do modelo ÷ erro da sazonal ingênua (abaixo de 1, erra menos)",
            "Model error ÷ seasonal naive error (below 1 means lower error)",
            "Error del modelo ÷ error de la estacional ingenua (por debajo de 1, menos error)",
        ),
        "formato_x": "numero",
        "formato_y": "numero",
        "series": [
            {
                "nome": NOMES_MODELOS["gbm"],
                "papel": "principal",
                "pontos": [[artigo, round(float(v), 4)] for artigo, v in tabela["razao_gbm"].items()],
                "rotulos": rotulos,
            }
        ],
        "leitura": _txt(
            f"Razão entre o erro do gradient boosting e o da sazonal ingênua, artigo por artigo, nos 8 horizontes. Os "
            f"maiores ganhos estão nas datas móveis e no Imposto de renda; em {pior} o intervalo contém 1, e é empate.",
            f"Ratio of gradient boosting error to seasonal naive error, article by article, over the 8 horizons. The "
            f"biggest gains come from moving holidays and income tax; for {ARTIGOS_TRADUZIDOS[pior]['en']} the "
            f"interval includes 1, so it is a tie.",
            f"Razón entre el error del gradient boosting y el de la estacional ingenua, artículo por artículo, en los 8 "
            f"horizontes. Las mayores ganancias están en las fechas móviles y en el impuesto sobre la renta; en "
            f"{ARTIGOS_TRADUZIDOS[pior]['es']} el intervalo contiene el 1, y es un empate.",
        ),
    }


def graficos_extras(r: Resultados) -> list[dict]:
    """As outras figuras do README como especificação de dados, do achado mais importante para o menos."""
    return [
        _grafico_erro_por_horizonte(r),
        _grafico_cobertura_95(r),
        _grafico_exemplo(r),
        _grafico_cobertura_por_ano(r),
        _grafico_erro_por_artigo(r),
    ]
