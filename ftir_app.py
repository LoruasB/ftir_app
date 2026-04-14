import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt


st.set_page_config(page_title="FTIR - Área de Banda")

st.title("🔬 Análise de FTIR (Múltiplos Arquivos)")

files = st.file_uploader(
    "Carregue seus arquivos CSV/TXT",
    type=["csv", "txt"],
    accept_multiple_files=True
)

delimiter = st.selectbox(
    "Selecione o separador de colunas do arquivo",
    options=[(",", "Vírgula (,)"), ("\t", "Tabulação (Tab)"), (";", "Ponto E Vírgula (;)")],
    format_func=lambda x: x[1]
)[0]

# Manter dados entre cliques
if "dados_todos" not in st.session_state:
    st.session_state["dados_todos"] = {}

if files:

    nomes_arquivos = [file.name for file in files]

    arquivo_escolhido = st.selectbox(
        "Escolha o arquivo para visualizar o gráfico",
        nomes_arquivos
    )

    st.subheader("Intervalo da banda")

    min_wn = st.number_input("Número de onda mínimo", value=0.0)
    max_wn = st.number_input("Número de onda máximo", value=4000.0)

    converter = st.checkbox("Converter %T para absorbância")

    resultados = []

    if st.button("Calcular áreas"):

        # Limpa dados antigos
        st.session_state["dados_todos"] = {}

        for file in files:
            try:
                # Leitura robusta
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
                wn = wn[mask_valid]
                absorb = absorb[mask_valid]

                # Garantir ordem crescente
                if wn.iloc[0] > wn.iloc[-1]:
                    wn = wn[::-1]
                    absorb = absorb[::-1]

                # Conversão
                if converter:
                    absorb = -np.log10(absorb / 100)

                # Corrigir intervalo 
                if min_wn > max_wn:
                    min_wn, max_wn = max_wn, min_wn

                mask = (wn >= min_wn) & (wn <= max_wn)
                wn_band = wn[mask]
                abs_band = absorb[mask]

                if len(wn_band) > 1:
                    baseline = np.linspace(
                        abs_band.iloc[0],
                        abs_band.iloc[-1],
                        len(abs_band)
                    )

                    abs_corr = abs_band - baseline

                    area = abs(simpson(abs_corr, wn_band))

                    st.session_state["dados_todos"][file.name] = (
                        wn, absorb, wn_band, abs_band, baseline
                    )
                else:
                    area = np.nan

                resultados.append({
                    "Arquivo": file.name,
                    "Área": area
                })

            except:
                resultados.append({
                    "Arquivo": file.name,
                    "Área": np.nan
                })

        df_resultados = pd.DataFrame(resultados)

        st.subheader("Resultados")
        st.write(df_resultados)

        csv = df_resultados.to_csv(
            index=False,
            sep=';',
            decimal='.'
        ).encode('utf-8')

        st.download_button(
            label="📥 Baixar resultados (CSV)",
            data=csv,
            file_name="resultados_ftir.csv",
            mime="text/csv"
        )

    # Botão do gráfico
    if st.button("Mostrar gráfico"):

        if arquivo_escolhido in st.session_state["dados_todos"]:

            wn, absorb, wn_band, abs_band, baseline = st.session_state["dados_todos"][arquivo_escolhido]

            fig, ax = plt.subplots()

            ax.plot(wn, absorb, label="Espectro")
            ax.plot(wn_band, baseline, '--', label="Baseline")

            ax.fill_between(
                wn_band,
                abs_band,
                baseline,
                alpha=0.3,
                label="Área integrada"
            )

            ax.set_xlabel("Número de onda (cm⁻¹)")
            ax.set_ylabel("Intensidade")
            ax.legend()

            st.subheader(f"Visualização: {arquivo_escolhido}")
            st.pyplot(fig)

        else:
            st.warning("Clique primeiro em 'Calcular áreas'")