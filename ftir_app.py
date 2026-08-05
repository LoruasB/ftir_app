import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import plotly.graph_objects as go
from scipy import sparse
from scipy.sparse.linalg import spsolve

st.set_page_config(page_title="FTIR - Área de Banda", layout="wide")

st.title("🔬 Análise de FTIR (Múltiplos Arquivos)")

files = st.file_uploader(
    "Carregue seus arquivos CSV/TXT",
    type=["csv", "txt"],
    accept_multiple_files=True
)

delimiter = st.selectbox(
    "Selecione o separador de colunas do arquivo",
    options=[(",", "Vírgula (,)"), ("\t", "Tabulação (Tab)"), (";", "Ponto e Vírgula (;)")],
    format_func=lambda x: x[1]
)[0]

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z

# Inicializa o session state para manter dados entre interações
if "dados_todos" not in st.session_state:
    st.session_state["dados_todos"] = {}
if "df_resultados" not in st.session_state:
    st.session_state["df_resultados"] = None

if files:
    nomes_arquivos = [file.name for file in files]

    arquivo_escolhido = st.selectbox(
        "Escolha o arquivo para visualizar o gráfico",
        nomes_arquivos
    )

    st.subheader("Intervalo de banda")
    col_min, col_max = st.columns(2)
    with col_min:
        min_wn = st.number_input("Número de onda mínimo", value=0.0)
    with col_max:
        max_wn = st.number_input("Número de onda máximo", value=4000.0)

    converter = st.checkbox("Converter %T para absorbância")

    st.subheader("Normalização do espectro")
    normalizar = st.checkbox("Normalizar espectro")
    wn_ref = 1700.0  # Valor padrão caso 'normalizar' não seja marcado
    if normalizar:
        wn_ref = st.number_input(
            "Número de onda para normalização (cm⁻¹)",
            value=1700.0
        )

    st.subheader("Correção de baseline")
    tipo_baseline = st.selectbox(
        "Escolha o tipo de baseline",
        ["Sem baseline", "Linear", "ALS"]
    )

    lam = 1e10
    p = 0.01
    if tipo_baseline == "ALS":
        col1, col2 = st.columns(2)
        with col1:
            lam = st.number_input(
                "Lambda (suavidade)",
                value=1e10,
                format="%.1e",
                step=1e10
            )
        with col2:
            p = st.number_input(
                "p (assimetria)",
                value=0.01,
                format="%.4f",
                step=0.005
            )

    # Processamento dos cálculos
    if st.button("Calcular áreas"):
        st.session_state["dados_todos"] = {}
        resultados = []

        for file in files:
            try:
                # Retorna ao início do arquivo caso ele já tenha sido lido antes
                file.seek(0)
                
                data = pd.read_csv(
                    file,
                    sep=delimiter,
                    engine='python',
                    decimal='.',
                    skiprows=2,
                    header=None
                )

                data = data.iloc[:, :2]
                data.columns = ["wn", "intensity"]

                wn = pd.to_numeric(data["wn"], errors='coerce')
                absorb = pd.to_numeric(data["intensity"], errors='coerce')

                mask_valid = wn.notna() & absorb.notna()
                wn = wn[mask_valid].reset_index(drop=True)
                absorb = absorb[mask_valid].reset_index(drop=True)

                if wn.iloc[0] > wn.iloc[-1]:
                    wn = wn[::-1].reset_index(drop=True)
                    absorb = absorb[::-1].reset_index(drop=True)

                if converter:
                    absorb = -np.log10(absorb / 100)

                if normalizar:
                    idx_ref = (np.abs(wn - wn_ref)).idxmin()
                    ref_value = absorb.iloc[idx_ref]

                    if ref_value != 0:
                        absorb = absorb / ref_value
                    else:
                        st.warning(f"{file.name}: valor de referência zero, não foi possível normalizar.")

                _min, _max = min(min_wn, max_wn), max(min_wn, max_wn)
                mask = (wn >= _min) & (wn <= _max)
                wn_band = wn[mask]
                abs_band = absorb[mask]

                if len(wn_band) > 1:
                    if tipo_baseline == "Sem baseline":
                        baseline = np.zeros_like(abs_band.values)
                    elif tipo_baseline == "Linear":
                        baseline = np.linspace(
                            abs_band.iloc[0],
                            abs_band.iloc[-1],
                            len(abs_band)
                        )
                    elif tipo_baseline == "ALS":
                        baseline = baseline_als(abs_band.values, lam=lam, p=p)

                    abs_corr = abs_band.values - baseline
                    area = abs(simpson(abs_corr, x=wn_band.values))

                    st.session_state["dados_todos"][file.name] = (
                        wn, absorb, wn_band, abs_band, baseline
                    )
                else:
                    area = np.nan

                resultados.append({
                    "Arquivo": file.name,
                    "Área": area
                })

            except Exception as e:
                st.error(f"Erro ao processar {file.name}: {str(e)}")
                resultados.append({
                    "Arquivo": file.name,
                    "Área": np.nan
                })

        st.session_state["df_resultados"] = pd.DataFrame(resultados)

    # Exibição dos resultados (se já calculados)
    if st.session_state["df_resultados"] is not None:
        df_resultados = st.session_state["df_resultados"]

        st.subheader("Resultados")
        st.dataframe(df_resultados)

        col_csv, col_txt = st.columns(2)
        with col_csv:
            csv = df_resultados.to_csv(index=False, sep=';', decimal='.').encode('utf-8')
            st.download_button(
                label="📥 Baixar resultados (CSV)",
                data=csv,
                file_name="resultados_ftir.csv",
                mime="text/csv"
            )
        with col_txt:
            txt = df_resultados.to_csv(index=False, sep='\t', decimal='.').encode('utf-8')
            st.download_button(
                label="📄 Baixar resultados (TXT)",
                data=txt,
                file_name="resultados_ftir.txt",
                mime="text/plain"
            )

    # Gráfico
    st.markdown("---")
    if st.button("Mostrar gráfico"):
        if arquivo_escolhido in st.session_state["dados_todos"]:
            wn, absorb, wn_band, abs_band, baseline = st.session_state["dados_todos"][arquivo_escolhido]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wn, y=absorb, mode='lines', name='Espectro'))

            if tipo_baseline != "Sem baseline":
                fig.add_trace(go.Scatter(
                    x=wn_band, y=baseline,
                    mode='lines', name=f'Baseline ({tipo_baseline})',
                    line=dict(dash='dash')
                ))

            fig.add_trace(go.Scatter(
                x=wn_band, y=abs_band,
                mode='lines', fill='tonexty',
                name='Área integrada', opacity=0.3
            ))

            fig.update_layout(
                xaxis_title="Número de onda (cm⁻¹)",
                yaxis_title="Intensidade normalizada" if normalizar else "Intensidade",
                xaxis=dict(autorange="reversed")
            )

            st.subheader(f"Visualização: {arquivo_escolhido}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Clique primeiro em 'Calcular áreas'")
