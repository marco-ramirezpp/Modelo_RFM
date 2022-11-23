import streamlit as st 
import pandas as pd 
import joblib 


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

modelo_RF = joblib_model = joblib.load('modelo_RF_221122')

add_selectbox = st.sidebar.selectbox('', ('Pagina principal', 'Modelo', 'Modelo csv'))

if add_selectbox == 'Pagina principal':
    st.image('cliente.png')
    st.title('Modelo de predicción de valor de cliente')
    st.text('Este modelo determina el valor de un cliente a partir de algunas caracteristicas observadas')

if add_selectbox == 'Modelo':
    st.title('Aquí puede realizar la prediccion de cada uno de los clientes')
    Age	= st.number_input('Edad',min_value=18, max_value=99, value=35)
    monto_gasto = st.number_input('Ingrese el monto de gasto promedio',min_value=0, max_value=2000)
    Recency = st.number_input('Numero de dias desde que el cliente no compra',min_value=0, max_value=365)
    Income = st.number_input('Ingreso anual del cliente',min_value=15000, max_value=200000)		
    frecuencia =  st.number_input('Ingrese el numero de items promedio al mes',min_value=0, max_value=150)

    input_dict = {'Age': Age, 'monto_gasto': monto_gasto,
                  'Recency': Recency, 'Income': Income,
                  'frecuencia': frecuencia}
    
    input_df = pd.DataFrame([input_dict])

    if st.button('Predicción'):
        salida = modelo_RF.predict(input_df)
        salida1 = int(salida)
        
        if salida1 == 3:
            st.success('Este cliente es de: Muy Alto Valor')
        if salida1 == 4:
            st.success('Este cliente es de: Alto Valor')
        if salida1 == 1:
            st.success('Este cliente es de: Mediano Valor')
        if salida1 == 0:
            st.success('Este cliente es de: Valor Regular')
        if salida1 == 2:
            st.success('Este cliente es de: Bajo Valor')
        
if add_selectbox == 'Modelo csv':
    st.title('Aquí puede realizar la prediccion en lote de los clientes')
    data = st.file_uploader('Archivo', type='csv')
    if data is not None:
        df = pd.read_csv(data)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        st.dataframe(df)

    if st.button('predecir'):
        df['Prediccion'] = modelo_RF.predict(df)
        st.dataframe(df)
        csv = convert_df(df)

        st.download_button(
            label="Descargar como CSV",
            data=csv,
            file_name='predicciones.csv',
            mime='text/csv',
        )
    
