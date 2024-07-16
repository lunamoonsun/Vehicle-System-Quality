import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Configuração da página, ocupando a pagina toda
st.set_page_config(
    page_title="Classificador de Veículos",
    layout="wide"
)

@st.cache_data # Cache para não precisar carregar o modelo toda vez que a página é recarregada
def load_data_and_model():
    data = pd.read_csv("car.csv", sep=",")

    # Tratamento dos dados
    encoder = OrdinalEncoder()

    # Transformando variaveis para catergoricas
    for col in data.columns.drop('class'):
        data[col] = data[col].astype('category')

    X_encoded = encoder.fit_transform(data.drop('class', axis=1))
    
    # Transformando a variavel target para categorica
    y = data['class'].astype('category').cat.codes

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # random_state é uma semente para garantir que a divisão seja sempre a mesma

    modelo = CategoricalNB()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    # accuracy_score é uma função que compara os valores reais com os valores previstos
    # retornando uma acurácia
    accuracy = accuracy_score(y_test, y_pred)

    return encoder, modelo, accuracy, data

encoder, modelo, accuracy, data = load_data_and_model()

st.title("Previsão de Qualidade de Veículos")
st.write(f"Acurácia do modelo: {accuracy: .2f}")

input_features = [
    st.selectbox("Preço: ", data['buying'].unique()),
    st.selectbox("Manutenção: ", data['maint'].unique()),
    st.selectbox("Portas: ", data['doors'].unique()),
    st.selectbox("Capacidade de passageiros: ", data['persons'].unique()),
    st.selectbox("Porta malas: ", data['lug_boot'].unique()),
    st.selectbox("Segurança: ", data['safety'].unique())
]

if st.button("Processar"):
    input_df = pd.DataFrame([input_features], columns=data.columns.drop('class'))
    input_encoder = encoder.transform(input_df)
    predict_encoded = modelo.predict(input_encoder)
    previsao = data['class'].astype('category').cat.categories[predict_encoded][0]
    st.header(f"O veículo é de qualidade: {previsao}")