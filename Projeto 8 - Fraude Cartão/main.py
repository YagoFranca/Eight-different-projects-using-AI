import encodings

import streamlit as st
import pandas as pd
import joblib

# Carrega o modelo treinado
model = joblib.load('../IA/Pickle/frauddetection.pkl')  # ajuste o caminho se necessário

# Supondo que este é seu DataFrame original (antes da codificação)
df = pd.read_csv("../Projeto 22 - Fraude Cartão/undersampled_data.csv")

# Extrai valores únicos das colunas categóricas
merchant_names = df['MerchantName'].astype(str).unique().tolist()
merchant_cities = df['MerchantCity'].astype(str).unique().tolist()
merchant_states = df['MerchantState'].astype(str).unique().tolist()

# Interface do Streamlit
st.title("Detecção de Fraude")

# Seletores usando os valores do CSV
merchant_name = st.selectbox("Nome do Estabelecimento", options=merchant_names)
merchant_city = st.selectbox("Cidade", options=merchant_cities)
merchant_state = st.selectbox("País", options=merchant_states)

# Converte as seleções para códigos (índices)
merchant_name_code = merchant_names.index(merchant_name)
merchant_city_code = merchant_cities.index(merchant_city)
merchant_state_code = merchant_states.index(merchant_state)

# Outros campos
year = st.number_input("Ano", min_value=2000, max_value=2100, value=2024)
month = st.number_input("Mês", min_value=1, max_value=12, value=1)
use_chip_label = st.selectbox("Usou Chip ?", options=["Sim", "Não"])
use_chip = 1 if use_chip_label == "Sim" else 0
amount = st.number_input("Valor da Transação", value=100)
mcc = st.number_input("MCC (Merchant Category Code)", value=0)


# Adicione isto antes da previsão para verificar os códigos gerados:
#print("Códigos convertidos:", merchant_name_code, merchant_city_code, merchant_state_code)


#print("Posição:", merchant_names[326])

# Verificação
if st.button("Verificar Fraude"):
    features = [[year, month, use_chip, amount, merchant_name_code, merchant_city_code, merchant_state_code, mcc]]
    prediction = model.predict(features)
    result = "🚨 Fraude Detectada!" if prediction[0] == 1 else "✅ Sem Fraude"
    st.subheader(f"Resultado: {result}")