import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from numpy import linalg as LA
import datetime as dt
from bcb import sgs

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Markowitz + Monte Carlo", layout="wide", page_icon="📈")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border-radius: 8px 8px 0 0;
        padding: 10px 24px; font-weight: 600; color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important; color: white !important;
    }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 18px 20px; text-align: center;
    }
    .metric-card .label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #58a6ff; margin-top: 4px; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .section-header {
        border-left: 4px solid #1f6feb; padding-left: 12px;
        margin: 24px 0 16px; font-size: 18px; font-weight: 700; color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Markowitz + Monte Carlo")
st.markdown("**Otimize sua carteira pela teoria de Markowitz e projete cenários futuros com simulação de Monte Carlo.**")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚙️ Parâmetros Gerais")

tickers_input = st.sidebar.text_input(
    "Tickers (vírgula, sem .SA)",
    "PETR4, VALE3, ITUB4, WEGE3, BBDC4, ABEV3"
)
tickers = [t.strip().upper() + ".SA" for t in tickers_input.split(",") if t.strip()]

data_inicio = st.sidebar.date_input("Data inicial (histórico)", dt.date(2022, 1, 1))
data_fim = st.sidebar.date_input("Data final (histórico)", dt.date.today())

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Markowitz")
n_carteiras = st.sidebar.number_input("Carteiras simuladas", 1000, 200000, 30000, 1000)

opcoes_indices = {
    "CDI": "CDI", "IPCA": "IPCA",
    "IFIX": "XFIX11.SA", "IDIV": "DIVO11.SA",
    "SMLL": "SMAL11.SA", "IBOV": "^BVSP",
    "S&P 500": "^GSPC", "IVVB11": "IVVB11.SA"
}
sgs_map = {"CDI": 12, "IPCA": 433}
escolhidos = st.sidebar.multiselect("Benchmarks", list(opcoes_indices.keys()), default=["CDI", "IBOV"])

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎲 Monte Carlo")
capital_inicial = st.sidebar.number_input("Capital inicial (R$)", 1000, 10_000_000, 10_000, 500)
numero_simulacoes = st.sidebar.number_input("Número de simulações", 1000, 50000, 10000, 1000)
anos_projetados = st.sidebar.slider("Horizonte (anos)", 1, 10, 3)
dias_projetados = 252 * anos_projetados

rodar = st.sidebar.button("🚀 Rodar Análise Completa", use_container_width=True)

# ============================================================
# MAIN LOGIC
# ============================================================
if not rodar:
    st.info("👈 Configure os parâmetros na barra lateral e clique em **Rodar Análise Completa**.")
    st.stop()

# ---------- Download de preços ----------
with st.spinner("📡 Baixando dados históricos..."):
    precos_raw = yf.download(tickers, start=data_inicio, end=data_fim, auto_adjust=False)["Adj Close"]
    if isinstance(precos_raw, pd.Series):
        precos_raw = precos_raw.to_frame()
    failed = [c for c in precos_raw.columns if precos_raw[c].dropna().empty]
    if failed:
        st.warning(f"Tickers sem dados ignorados: {failed}")
        precos_raw = precos_raw.drop(columns=failed)
        tickers = [t for t in tickers if t not in failed]
    if precos_raw.empty:
        st.error("Nenhum dado retornado. Verifique os tickers.")
        st.stop()

precos = precos_raw.copy()
retornos = precos.pct_change().dropna()
media_retornos = retornos.mean()
matriz_cov = retornos.cov()
n_ativos = len(tickers)

# ============================================================
# ABA 1: MARKOWITZ
# ============================================================
tab1, tab2 = st.tabs(["🏆 Otimização de Markowitz", "🎲 Simulação de Monte Carlo"])

with tab1:
    st.markdown('<div class="section-header">Fronteira Eficiente de Markowitz</div>', unsafe_allow_html=True)

    with st.spinner("⚙️ Simulando carteiras aleatórias..."):
        ret_sim = np.zeros(n_carteiras)
        vol_sim = np.zeros(n_carteiras)
        sharpe_sim = np.zeros(n_carteiras)
        pesos_sim = np.zeros((n_carteiras, n_ativos))

        for k in range(n_carteiras):
            p = np.random.random(n_ativos)
            p /= p.sum()
            pesos_sim[k] = p
            ret_sim[k] = np.sum(media_retornos * p) * 252
            vol_sim[k] = np.sqrt(p @ (matriz_cov * 252) @ p)
            sharpe_sim[k] = ret_sim[k] / (vol_sim[k] if vol_sim[k] != 0 else 1e-9)

        idx_max = sharpe_sim.argmax()
        pesos_otimos = pesos_sim[idx_max]
        ret_max = ret_sim[idx_max]
        vol_max = vol_sim[idx_max]
        sharpe_max = sharpe_sim[idx_max]

    # Métricas resumo
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Retorno Anual Esperado</div><div class="value">{ret_max*100:.2f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Volatilidade Anual</div><div class="value">{vol_max*100:.2f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">Índice de Sharpe</div><div class="value">{sharpe_max:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Tabela de pesos
    st.markdown('<div class="section-header">Alocação da Carteira Ótima</div>', unsafe_allow_html=True)
    df_pesos = pd.DataFrame({
        "Ativo": [t.replace(".SA", "") for t in tickers],
        "Peso (%)": pesos_otimos * 100
    }).sort_values("Peso (%)", ascending=False).reset_index(drop=True)
    st.dataframe(df_pesos.style.format({"Peso (%)": "{:.2f}%"}).bar(subset=["Peso (%)"], color="#1f6feb"), use_container_width=True)

    # Fronteira eficiente
    with st.spinner("📐 Calculando fronteira eficiente..."):
        eixo_y_fe = np.linspace(ret_sim.min(), ret_sim.max(), 50)

        def ret_fn(w): return np.sum(media_retornos * w) * 252
        def vol_fn(w): return np.sqrt(w @ (matriz_cov * 252) @ w)

        p0 = [1 / n_ativos] * n_ativos
        bounds = [(0, 1)] * n_ativos
        eixo_x_fe = []
        for r_alvo in eixo_y_fe:
            res = minimize(vol_fn, p0, method="SLSQP", bounds=bounds, constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w: ret_fn(w) - r_alvo},
            ])
            eixo_x_fe.append(res.fun)

    fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
    ax1.set_facecolor("#0d1117")
    sc = ax1.scatter(vol_sim, ret_sim, c=sharpe_sim, cmap="plasma", alpha=0.5, s=3)
    ax1.scatter(vol_max, ret_max, c="#00ff88", s=180, zorder=5, label="Maior Sharpe", marker="*")
    ax1.plot(eixo_x_fe, eixo_y_fe, color="#58a6ff", linewidth=2.5, label="Fronteira Eficiente")
    ax1.set_xlabel("Volatilidade Esperada", color="#8b949e")
    ax1.set_ylabel("Retorno Esperado", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")
    plt.colorbar(sc, ax=ax1, label="Índice de Sharpe").ax.yaxis.label.set_color("#8b949e")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    ax1.grid(alpha=0.1, color="#30363d")
    st.pyplot(fig1)
    plt.close()

    # Comparativo histórico
    st.markdown('<div class="section-header">Desempenho Histórico vs Benchmarks</div>', unsafe_allow_html=True)
    with st.spinner("📊 Carregando benchmarks..."):
        carteira_acc = (1 + retornos @ pesos_otimos).cumprod()
        comparativo = pd.DataFrame({"Carteira Ótima": carteira_acc})

        for nome in ["CDI", "IPCA"]:
            if nome not in escolhidos:
                continue
            try:
                sid = sgs_map[nome]
                df_idx = sgs.get({nome: sid}, start=carteira_acc.index.min())
                if df_idx.empty:
                    continue
                serie = df_idx.iloc[:, 0].astype(float) / 100.0
                serie.index = pd.to_datetime(serie.index)
                serie_acc = (1 + serie).cumprod().reindex(carteira_acc.index, method="ffill")
                comparativo[nome] = serie_acc
            except Exception as e:
                st.warning(f"Erro {nome}: {e}")

        for nome in escolhidos:
            if nome in sgs_map:
                continue
            ticker_bm = opcoes_indices.get(nome)
            if not ticker_bm:
                continue
            try:
                df_bm = yf.download(ticker_bm, start=carteira_acc.index.min(),
                                    end=carteira_acc.index.max(), auto_adjust=False)["Adj Close"]
                if isinstance(df_bm, pd.Series):
                    df_bm = df_bm.to_frame()
                if not df_bm.empty:
                    acc = (1 + df_bm.pct_change().dropna()).cumprod()
                    acc = acc.reindex(carteira_acc.index, method="ffill")
                    comparativo[nome] = acc.iloc[:, 0] if isinstance(acc, pd.DataFrame) else acc
            except Exception as e:
                st.warning(f"Erro {nome}: {e}")

    cores = {
        "Carteira Ótima": "#58a6ff", "CDI": "#3fb950", "IPCA": "#f78166",
        "IFIX": "#56d364", "IDIV": "#8b949e", "SMLL": "#bc8cff",
        "IBOV": "#ff7b72", "S&P 500": "#d2a8ff", "IVVB11": "#ffa657"
    }
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax2.set_facecolor("#0d1117")
    for col in comparativo.columns:
        lw = 2.5 if col == "Carteira Ótima" else 1.8
        ax2.plot(comparativo.index, (comparativo[col] - 1) * 100,
                 label=col, color=cores.get(col, "#ffffff"), linewidth=lw, alpha=0.9)
    ax2.set_ylabel("Rentabilidade Acumulada (%)", color="#8b949e")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    ax2.grid(alpha=0.1, color="#30363d")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
    st.pyplot(fig2)
    plt.close()

    retorno_final = (comparativo.iloc[-1] - 1) * 100
    st.dataframe(retorno_final.to_frame("Retorno Acumulado (%)").style.format("{:.2f}%"), use_container_width=True)

# ============================================================
# ABA 2: MONTE CARLO (usa pesos de Markowitz)
# ============================================================
with tab2:
    st.markdown('<div class="section-header">Simulação de Monte Carlo com Pesos de Markowitz</div>', unsafe_allow_html=True)
    st.info(f"Usando os **pesos ótimos de Markowitz** para projetar R$ {capital_inicial:,.2f} por **{anos_projetados} anos** ({dias_projetados} dias úteis) com **{numero_simulacoes:,} simulações**.")

    with st.spinner("🎲 Executando simulações..."):
        retorno_medio = retornos.mean(axis=0).to_numpy()
        matriz_retorno_medio = retorno_medio * np.ones(shape=(dias_projetados, n_ativos))

        try:
            L = LA.cholesky(matriz_cov.values)
        except LA.LinAlgError:
            # fallback: regularização
            cov_reg = matriz_cov.values + np.eye(n_ativos) * 1e-8
            L = LA.cholesky(cov_reg)

        retornos_carteira = np.zeros((dias_projetados, numero_simulacoes))
        montante_final = np.zeros(numero_simulacoes)

        for s in range(numero_simulacoes):
            Rpdf = np.random.normal(size=(dias_projetados, n_ativos))
            retornos_sinteticos = matriz_retorno_medio + np.inner(Rpdf, L)
            retornos_carteira[:, s] = np.cumprod(np.inner(pesos_otimos, retornos_sinteticos) + 1) * capital_inicial
            montante_final[s] = retornos_carteira[-1, s]

    # Métricas Monte Carlo
    med_50 = np.median(montante_final)
    med_95 = np.percentile(montante_final, 5)
    med_99 = np.percentile(montante_final, 1)
    prob_lucro = np.sum(montante_final > capital_inicial) / numero_simulacoes

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Mediana (50%)</div><div class="value">R$ {med_50:,.0f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Piso 95% confiança</div><div class="value">R$ {med_95:,.0f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">Piso 99% confiança</div><div class="value">R$ {med_99:,.0f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="label">Prob. de Lucro</div><div class="value">{prob_lucro:.1%}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ---- Gráfico das trajetórias ----
    st.markdown('<div class="section-header">Trajetórias Simuladas</div>', unsafe_allow_html=True)

    def fmt_brl(x, _): return f"R${x:,.0f}".replace(",", ".")

    fig3, ax3 = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
    ax3.set_facecolor("#0d1117")

    # Plota amostra de trajetórias
    amostra = min(500, numero_simulacoes)
    idx_amostra = np.random.choice(numero_simulacoes, amostra, replace=False)
    for i in idx_amostra:
        ax3.plot(retornos_carteira[:, i], linewidth=0.4, alpha=0.15, color="#58a6ff")

    # Percentis
    p50 = np.percentile(retornos_carteira, 50, axis=1)
    p05 = np.percentile(retornos_carteira, 5, axis=1)
    p95 = np.percentile(retornos_carteira, 95, axis=1)

    ax3.plot(p50, color="#00ff88", linewidth=2.5, label="Mediana (50%)", zorder=5)
    ax3.plot(p05, color="#ff7b72", linewidth=1.8, linestyle="--", label="Percentil 5%", zorder=5)
    ax3.plot(p95, color="#ffa657", linewidth=1.8, linestyle="--", label="Percentil 95%", zorder=5)
    ax3.fill_between(range(dias_projetados), p05, p95, alpha=0.1, color="#58a6ff")
    ax3.axhline(capital_inicial, color="#8b949e", linewidth=1, linestyle=":", label="Capital Inicial")

    ax3.set_xlabel("Dias Úteis", color="#8b949e")
    ax3.set_ylabel("Valor da Carteira (R$)", color="#8b949e")
    ax3.tick_params(colors="#8b949e")
    ax3.yaxis.set_major_formatter(FuncFormatter(fmt_brl))
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")
    ax3.grid(alpha=0.1, color="#30363d")
    ax3.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)

    # Rótulos no eixo X com anos
    tick_pos = [i * 252 for i in range(anos_projetados + 1)]
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels([f"Ano {i}" for i in range(anos_projetados + 1)], color="#8b949e")
    st.pyplot(fig3)
    plt.close()

    # ---- Histograma dos montantes finais ----
    st.markdown('<div class="section-header">Distribuição dos Montantes Finais</div>', unsafe_allow_html=True)

    fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax4.set_facecolor("#0d1117")
    ax4.hist(montante_final, bins=120, color="#1f6feb", alpha=0.75, edgecolor="none")
    ax4.axvline(med_50, color="#00ff88", linewidth=2, label=f"Mediana: R$ {med_50:,.0f}")
    ax4.axvline(med_95, color="#ff7b72", linewidth=2, linestyle="--", label=f"P5 (95% conf.): R$ {med_95:,.0f}")
    ax4.axvline(capital_inicial, color="#8b949e", linewidth=1.5, linestyle=":", label=f"Capital inicial: R$ {capital_inicial:,.0f}")
    ax4.set_xlabel("Montante Final (R$)", color="#8b949e")
    ax4.set_ylabel("Frequência", color="#8b949e")
    ax4.tick_params(colors="#8b949e")
    ax4.xaxis.set_major_formatter(FuncFormatter(fmt_brl))
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")
    ax4.grid(alpha=0.1, color="#30363d")
    ax4.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
    st.pyplot(fig4)
    plt.close()

    # Resumo textual
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin-top:16px;line-height:1.8">
    <b style="color:#58a6ff">📋 Interpretação dos Resultados</b><br><br>
    Ao investir <b>R$ {capital_inicial:,.2f}</b> com a carteira ótima de Markowitz por <b>{anos_projetados} anos</b>:<br><br>
    • Com <b>50% de probabilidade</b>, o montante final superará <b>R$ {med_50:,.2f}</b><br>
    • Com <b>95% de probabilidade</b>, o montante final superará <b>R$ {med_95:,.2f}</b><br>
    • Com <b>99% de probabilidade</b>, o montante final superará <b>R$ {med_99:,.2f}</b><br>
    • A probabilidade de obter lucro é de <b>{prob_lucro:.1%}</b>
    </div>
    """, unsafe_allow_html=True)