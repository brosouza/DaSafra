import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# CONFIGURAÇÃO E DADOS
# ============================================================

st.set_page_config(
    page_title="DaSafra | Inteligência Agrícola",
    page_icon="🌽",
    layout="wide",
)

CAMINHO_BASE = Path("data/gold/datasets")
CAMINHO_HISTORICO = CAMINHO_BASE / "dasafra_estado_ano.parquet"
CAMINHO_MODELO = CAMINHO_BASE / "dasafra_resultados_modelo.parquet"


@st.cache_data
def carregar_dados():
    df_historico = pd.read_parquet(CAMINHO_HISTORICO)
    df_modelo = pd.read_parquet(CAMINHO_MODELO)
    return df_historico, df_modelo


df_historico, df_modelo = carregar_dados()


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def configurar_grafico(fig, altura=400, margem=None):
    if margem is None:
        margem = dict(l=10, r=10, t=35, b=10)

    fig.update_layout(
        height=altura,
        margin=margem,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAEAEA"),
    )
    return fig


def calcular_media_ponderada(grupo, coluna):
    dados_validos = grupo.dropna(subset=[coluna, "area_colhida_total"])

    if dados_validos.empty:
        return np.nan

    peso_total = dados_validos["area_colhida_total"].sum()
    if peso_total == 0:
        return np.nan

    return (
        dados_validos[coluna] * dados_validos["area_colhida_total"]
    ).sum() / peso_total


def preparar_clima_nacional(df):
    variaveis = [
        "precipitacao_anual_mm",
        "temperatura_media_c",
        "umidade_media_pct",
    ]

    return (
        df.groupby("ano")
        .apply(
            lambda grupo: pd.Series({
                coluna: calcular_media_ponderada(grupo, coluna)
                for coluna in variaveis
            }),
            include_groups=False,
        )
        .reset_index()
    )


def criar_grafico_relacao(df, variavel, titulo, cor, unidade):
    correlacao = df[variavel].corr(df["produtividade_kg_ha"])

    fig = px.scatter(
        df,
        x=variavel,
        y="produtividade_kg_ha",
        hover_data=["sigla_uf", "ano"],
        labels={
            variavel: unidade,
            "produtividade_kg_ha": "Produtividade (kg/ha)",
        },
        title=titulo,
        opacity=0.65,
    )

    # Linha de tendência linear descritiva
    coeficientes = np.polyfit(
        df[variavel],
        df["produtividade_kg_ha"],
        1,
    )
    funcao_tendencia = np.poly1d(coeficientes)
    valores_x = np.linspace(df[variavel].min(), df[variavel].max(), 100)

    fig.add_scatter(
        x=valores_x,
        y=funcao_tendencia(valores_x),
        mode="lines",
        name="Tendência linear",
        line=dict(color=cor, width=3),
    )

    fig.update_traces(
        marker=dict(color=cor, size=7),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAEAEA"),
        xaxis=dict(title=unidade, gridcolor="rgba(150,150,150,0.15)"),
        yaxis=dict(
            title="Produtividade (kg/ha)",
            gridcolor="rgba(150,150,150,0.15)",
        ),
        showlegend=False,
    )

    return fig, correlacao


# ============================================================
# SEÇÃO 1 — VISÃO GERAL
# ============================================================

def exibir_visao_geral(df):
    st.title("🌽 DaSafra")
    st.subheader("Inteligência climática para a agricultura familiar")
    st.markdown(
        "Acompanhe indicadores de produtividade agrícola, condições climáticas "
        "e identificação de situações de baixa produtividade do milho nos estados brasileiros."
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Estados monitorados", df["sigla_uf"].nunique())

    with col2:
        st.metric(
            "Período histórico",
            f'{df["ano"].min()}–{df["ano"].max()}',
        )

    with col3:
        st.metric(
            "Produtividade média",
            f'{df["produtividade_kg_ha"].mean():,.0f} kg/ha',
        )

    with col4:
        st.metric("Registros avaliados pelo ML", len(df_modelo))

    st.divider()
    st.subheader("Produtividade agrícola por estado")

    anos_disponiveis = sorted(df["ano"].unique(), reverse=True)
    ano_selecionado = st.selectbox("Selecione o ano", anos_disponiveis)

    df_mapa = df[df["ano"] == ano_selecionado].copy()

    fig_mapa = px.choropleth(
        df_mapa,
        geojson="https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/brazil-states.geojson",
        locations="sigla_uf",
        featureidkey="properties.sigla",
        color="produtividade_kg_ha",
        color_continuous_scale="Greens",
        scope="south america",
        hover_name="sigla_uf",
        hover_data={
            "produtividade_kg_ha": ":,.0f",
            "quantidade_produzida_total": ":,.0f",
            "sigla_uf": False,
        },
        labels={
            "produtividade_kg_ha": "Produtividade (kg/ha)",
            "quantidade_produzida_total": "Produção (toneladas)",
        },
        title=f"Produtividade do milho em {ano_selecionado}",
    )

    fig_mapa.update_geos(
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
        center={"lat": -14.5, "lon": -52},
        lonaxis_range=[-75, -32],
        lataxis_range=[-35, 6],
    )

    fig_mapa.update_coloraxes(
        colorbar=dict(
            title="Produtividade (kg/ha)",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.05,
            yanchor="bottom",
            len=0.4,
            thickness=14,
        )
    )

    fig_mapa.update_layout(
        height=850,
        margin=dict(l=0, r=0, t=100, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAEAEA"),
    )

    st.plotly_chart(
        fig_mapa,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )

    st.divider()
    st.subheader(f"Ranking de produtividade por estado — {ano_selecionado}")

    df_ranking = (
        df_mapa[["sigla_uf", "produtividade_kg_ha"]]
        .dropna()
        .sort_values("produtividade_kg_ha", ascending=False)
    )

    df_top5 = df_ranking.head(5).sort_values(
        "produtividade_kg_ha", ascending=True
    )
    df_bottom5 = df_ranking.tail(5).sort_values(
        "produtividade_kg_ha", ascending=False
    )

    col_top, col_bottom = st.columns(2)

    with col_top:
        st.markdown("##### Maiores produtividades")
        fig_top = px.bar(
            df_top5,
            x="produtividade_kg_ha",
            y="sigla_uf",
            orientation="h",
            text_auto=",.0f",
            color="produtividade_kg_ha",
            color_continuous_scale="Greens",
            labels={
                "sigla_uf": "Estado",
                "produtividade_kg_ha": "Produtividade (kg/ha)",
            },
        )
        fig_top.update_layout(
            height=350,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAEAEA"),
            xaxis_title="Produtividade (kg/ha)",
            yaxis_title="",
        )
        st.plotly_chart(
            fig_top, use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_bottom:
        st.markdown("##### Menores produtividades")
        fig_bottom = px.bar(
            df_bottom5,
            x="produtividade_kg_ha",
            y="sigla_uf",
            orientation="h",
            text_auto=",.0f",
            color="produtividade_kg_ha",
            color_continuous_scale="Greens_r",
            labels={
                "sigla_uf": "Estado",
                "produtividade_kg_ha": "Produtividade (kg/ha)",
            },
        )
        fig_bottom.update_layout(
            height=350,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAEAEA"),
            xaxis_title="Produtividade (kg/ha)",
            yaxis_title="",
        )
        st.plotly_chart(
            fig_bottom, use_container_width=True,
            config={"displayModeBar": False},
        )


# ============================================================
# SEÇÃO 2 — HISTÓRICO AGRÍCOLA
# ============================================================

def exibir_historico(df):
    st.title("Evolução histórica da produtividade")
    st.markdown(
        "Acompanhe a evolução da produtividade nacional do milho entre 2001 e 2024."
    )

    df_evolucao = (
        df.groupby("ano", as_index=False)
        .agg({
            "quantidade_produzida_total": "sum",
            "area_colhida_total": "sum",
        })
    )

    df_evolucao["produtividade_kg_ha"] = (
        df_evolucao["quantidade_produzida_total"] * 1000
        / df_evolucao["area_colhida_total"]
    )

    fig = px.line(
        df_evolucao,
        x="ano",
        y="produtividade_kg_ha",
        markers=True,
        labels={
            "ano": "Ano",
            "produtividade_kg_ha": "Produtividade (kg/ha)",
        },
        title="Produtividade nacional do milho — 2001 a 2024",
    )
    fig.update_traces(
        line=dict(width=3, color="#3182CE"),
        marker=dict(size=7),
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAEAEA"),
        xaxis=dict(
            dtick=1,
            showgrid=False,
            range=[2000.5, 2024.5],
            title="Ano",
        ),
        yaxis=dict(
            title="Produtividade (kg/ha)",
            gridcolor="rgba(150,150,150,0.15)",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False},
    )

    st.divider()
    st.subheader("Explorar por estado")

    estados = sorted(df["sigla_uf"].dropna().unique())
    estado = st.selectbox("Selecione um estado", estados)

    df_estado = df[df["sigla_uf"] == estado].sort_values("ano")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Produtividade média no período",
            f'{df_estado["produtividade_kg_ha"].mean():,.0f} kg/ha',
        )
    with col2:
        st.metric("Anos disponíveis", len(df_estado))

    fig_estado = px.line(
        df_estado,
        x="ano",
        y="produtividade_kg_ha",
        markers=True,
        labels={
            "ano": "Ano",
            "produtividade_kg_ha": "Produtividade (kg/ha)",
        },
        title=f"Evolução da produtividade — {estado}",
    )
    fig_estado.update_traces(
        line=dict(color="#2CA25F", width=3),
        marker=dict(size=7),
    )
    fig_estado.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAEAEA"),
        xaxis=dict(dtick=1, showgrid=False, range=[2000.5, 2024.5]),
        yaxis=dict(gridcolor="rgba(150,150,150,0.15)"),
    )
    st.plotly_chart(
        fig_estado, use_container_width=True,
        config={"displayModeBar": False},
    )


# ============================================================
# SEÇÃO 3 — INDICADORES CLIMÁTICOS
# ============================================================

def exibir_indicadores_climaticos(df):
    st.title("Indicadores climáticos")
    st.markdown(
        "Evolução histórica das condições climáticas associadas à produção de milho."
    )

    df_clima = preparar_clima_nacional(df)

    configuracoes = [
        (
            "Precipitação anual",
            "precipitacao_anual_mm",
            "Precipitação (mm)",
            "#3182CE",
        ),
        (
            "Temperatura média",
            "temperatura_media_c",
            "Temperatura (°C)",
            "#E45756",
        ),
        (
            "Umidade relativa média",
            "umidade_media_pct",
            "Umidade (%)",
            "#9467BD",
        ),
    ]

    colunas = st.columns(3)

    for coluna, (titulo, variavel, unidade, cor) in zip(colunas, configuracoes):
        with coluna:
            st.markdown(f"#### {titulo}")
            fig = px.line(
                df_clima,
                x="ano",
                y=variavel,
                markers=True,
                labels={"ano": "Ano", variavel: unidade},
            )
            fig.update_traces(
                line=dict(color=cor, width=2.5),
                marker=dict(size=5),
            )
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAEAEA"),
                xaxis=dict(
                    range=[2000.5, 2024.5],
                    dtick=5,
                    showgrid=False,
                ),
                yaxis=dict(gridcolor="rgba(150,150,150,0.15)"),
                showlegend=False,
            )
            st.plotly_chart(
                fig, use_container_width=True,
                config={"displayModeBar": False},
            )

    st.divider()
    st.subheader("Relação entre clima e produtividade")
    st.markdown(
        "Análise exploratória da associação entre indicadores climáticos "
        "e produtividade estadual do milho. Cada ponto representa um estado "
        "em determinado ano."
    )

    variaveis = [
        "precipitacao_anual_mm",
        "temperatura_media_c",
        "umidade_media_pct",
        "produtividade_kg_ha",
    ]
    df_relacao = df.dropna(subset=variaveis).copy()

    relacoes = [
        (
            "precipitacao_anual_mm",
            "Precipitação x Produtividade",
            "#3182CE",
            "Precipitação (mm)",
        ),
        (
            "temperatura_media_c",
            "Temperatura x Produtividade",
            "#E45756",
            "Temperatura média (°C)",
        ),
        (
            "umidade_media_pct",
            "Umidade x Produtividade",
            "#9467BD",
            "Umidade relativa (%)",
        ),
    ]

    colunas = st.columns(3)
    for coluna, (variavel, titulo, cor, unidade) in zip(colunas, relacoes):
        with coluna:
            fig, correlacao = criar_grafico_relacao(
                df_relacao, variavel, titulo, cor, unidade
            )
            st.plotly_chart(
                fig, use_container_width=True,
                config={"displayModeBar": False},
            )
            st.metric("Correlação de Pearson", f"{correlacao:.3f}")

    st.caption(
        f"Análise baseada em {len(df_relacao)} registros estado-ano "
        "com dados completos. As correlações representam associações "
        "estatísticas e não demonstram relações de causalidade."
    )


# ============================================================
# SEÇÃO 4 — ANÁLISE DE RISCO
# ============================================================

def exibir_analise_risco(df):
    st.title("Análise de risco")
    st.markdown(
        "Classificação retrospectiva de risco de baixa produtividade realizada "
        "pelo modelo Random Forest para o conjunto de teste (2022–2024)."
    )

    y_real = df["risco_baixa_produtividade"].astype(int)
    y_previsto = df["risco_previsto"].astype(int)
    y_probabilidade = df["probabilidade_risco"]

    acuracia = accuracy_score(y_real, y_previsto)
    acuracia_balanceada = balanced_accuracy_score(y_real, y_previsto)
    precisao = precision_score(y_real, y_previsto, zero_division=0)
    recall = recall_score(y_real, y_previsto, zero_division=0)
    f1 = f1_score(y_real, y_previsto, zero_division=0)
    auc = roc_auc_score(y_real, y_probabilidade)
    average_precision = average_precision_score(y_real, y_probabilidade)

    matriz = confusion_matrix(y_real, y_previsto)
    verdadeiros_negativos = matriz[0, 0]
    falsos_positivos = matriz[0, 1]
    falsos_negativos = matriz[1, 0]
    verdadeiros_positivos = matriz[1, 1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia balanceada", f"{acuracia_balanceada:.1%}")
    col2.metric("Precisão", f"{precisao:.1%}")
    col3.metric("Recall de risco", f"{recall:.1%}")
    col4.metric("F1-score", f"{f1:.1%}")

    st.divider()
    col_matriz, col_metricas = st.columns([1.2, 1])

    with col_matriz:
        st.subheader("Matriz de confusão")
        fig_matriz = go.Figure(
            data=go.Heatmap(
                z=matriz,
                x=["Sem risco", "Com risco"],
                y=["Sem risco", "Com risco"],
                text=matriz,
                texttemplate="%{text}",
                textfont=dict(size=20),
                colorscale="Greens",
                showscale=False,
            )
        )
        fig_matriz.update_layout(
            height=400,
            xaxis_title="Classificação prevista",
            yaxis_title="Classificação real",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAEAEA"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(
            fig_matriz, use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_metricas:
        st.subheader("Desempenho complementar")
        st.metric("ROC-AUC", f"{auc:.3f}")
        st.metric("Average Precision", f"{average_precision:.3f}")
        st.metric("Estados-ano com alerta", int(y_previsto.sum()))
        st.metric(
            "Riscos identificados corretamente",
            f"{verdadeiros_positivos} de {int(y_real.sum())}",
        )

    st.divider()
    st.subheader("Interpretação das classificações")
    col_acertos, col_erros = st.columns(2)

    with col_acertos:
        st.markdown("##### Classificações corretas")
        st.write(f"Sem risco identificado corretamente: **{verdadeiros_negativos}**")
        st.write(f"Risco identificado corretamente: **{verdadeiros_positivos}**")

    with col_erros:
        st.markdown("##### Classificações incorretas")
        st.write(f"Falsos alertas: **{falsos_positivos}**")
        st.write(f"Riscos não identificados: **{falsos_negativos}**")

    st.divider()
    st.subheader("Resultados por estado")

    colunas_exibicao = [
        "sigla_uf",
        "ano",
        "produtividade_kg_ha",
        "media_historica_3_anos",
        "probabilidade_risco",
        "classificacao_risco",
    ]
    df_exibicao = df[colunas_exibicao].copy()
    df_exibicao["probabilidade_risco"] = (
        df_exibicao["probabilidade_risco"] * 100
    ).round(1)
    df_exibicao["produtividade_kg_ha"] = (
        df_exibicao["produtividade_kg_ha"].round(0)
    )
    df_exibicao["media_historica_3_anos"] = (
        df_exibicao["media_historica_3_anos"].round(0)
    )
    df_exibicao = df_exibicao.rename(
        columns={
            "sigla_uf": "Estado",
            "ano": "Ano",
            "produtividade_kg_ha": "Produtividade (kg/ha)",
            "media_historica_3_anos": "Média histórica (kg/ha)",
            "probabilidade_risco": "Probabilidade de risco (%)",
            "classificacao_risco": "Classificação",
        }
    )

    st.dataframe(
        df_exibicao.sort_values(
            "Probabilidade de risco (%)", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Avaliação retrospectiva realizada sobre 74 registros do conjunto "
        "de teste (2022–2024). O modelo utiliza indicadores climáticos anuais "
        "e não representa uma previsão antecipada da safra."
    )


# ============================================================
# SEÇÃO 5 — METODOLOGIA
# ============================================================

def exibir_metodologia():
    st.title("Metodologia")
    st.markdown(
        "Esta seção apresenta as fontes de dados, o processamento realizado "
        "e os limites de interpretação da solução DaSafra."
    )

    with st.expander("Fontes de dados", expanded=True):
        st.markdown(
            """
            **PAM — Produção Agrícola Municipal (IBGE)**  
            Dados de área colhida e quantidade produzida de milho em grão,
            por município e ano.

            **INMET — Instituto Nacional de Meteorologia**  
            Dados meteorológicos de estações automáticas, consolidados em
            indicadores anuais de precipitação, temperatura e umidade.
            """
        )

    with st.expander("Período e abrangência"):
        st.markdown(
            """
            - Período histórico: **2001 a 2024**.
            - Abrangência geográfica: **27 unidades da Federação**.
            - Cultura analisada: **milho em grão**.
            - Unidade de análise do dashboard e do modelo: **estado-ano**.
            """
        )

    with st.expander("Construção do indicador de produtividade"):
        st.markdown(
            """
            A produtividade estadual é calculada pela razão entre a quantidade
            produzida e a área colhida, ponderando o resultado pela produção
            agrícola agregada:

            **Produtividade (kg/ha) = produção (toneladas) × 1.000 / área colhida (ha)**
            """
        )

    with st.expander("Construção do target de risco"):
        st.markdown(
            """
            O target identifica anos em que a produtividade estadual ficou
            pelo menos 10% abaixo da média dos três anos anteriores.

            **Risco = produtividade do ano < 90% da média histórica dos três anos anteriores.**
            """
        )

    with st.expander("Modelo de Machine Learning"):
        st.markdown(
            """
            - Algoritmo: **Random Forest**.
            - Problema: classificação binária de baixa produtividade.
            - Variáveis: indicadores climáticos anuais, anomalias climáticas,
              ano e unidade da Federação.
            - Conjunto de teste: **2022 a 2024**.
            """
        )

    with st.expander("Limitações e cuidados de interpretação"):
        st.markdown(
            """
            - A PAM não diferencia a produção da agricultura familiar da
              produção comercial. Portanto, os dados não representam
              exclusivamente agricultores familiares.
            - A associação entre clima e produtividade não demonstra causalidade.
            - A associação entre município e estação meteorológica foi feita
              pela estação mais próxima do centroide municipal, uma aproximação.
            - Os indicadores climáticos são agregados anuais e não representam
              necessariamente as condições durante cada etapa do ciclo da cultura.
            - O modelo utiliza indicadores climáticos do próprio ano analisado.
              Portanto, sua avaliação é retrospectiva e não constitui uma
              previsão antecipada da safra.
            - Os resultados do modelo devem ser interpretados como apoio
              exploratório à análise, não como recomendação agronômica ou
              certificação de risco.
            """
        )


# ============================================================
# NAVEGAÇÃO
# ============================================================

st.sidebar.title("🌽 DaSafra")
st.sidebar.caption("Inteligência climática para a agricultura")
st.sidebar.divider()

secao = st.sidebar.radio(
    "Navegação",
    [
        "Visão geral",
        "Histórico agrícola",
        "Indicadores climáticos",
        "Análise de risco",
        "Metodologia",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Milho em grão | Brasil | 2001–2024")

if secao == "Visão geral":
    exibir_visao_geral(df_historico)

elif secao == "Histórico agrícola":
    exibir_historico(df_historico)

elif secao == "Indicadores climáticos":
    exibir_indicadores_climaticos(df_historico)

elif secao == "Análise de risco":
    exibir_analise_risco(df_modelo)

elif secao == "Metodologia":
    exibir_metodologia()
