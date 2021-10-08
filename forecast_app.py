import streamlit as st
import pandas as pd
import numpy as np
import os
from dateutil.parser import parse
from datetime import datetime
from darts import TimeSeries
from darts.models import ExponentialSmoothing, AutoARIMA, ARIMA
from darts.models import VARIMA, NBEATSModel, RNNModel, TransformerModel, BlockRNNModel, TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape

model_dict = {
    'Exponential Smoothing': ExponentialSmoothing(),
    'Auto ARIMA': AutoARIMA(),
    'ARIMA': ARIMA(),
    'RNNModel': RNNModel(input_chunk_length=24),
    'BlockRNNModel': BlockRNNModel(input_chunk_length=24, output_chunk_length=12),
    'NBEATSModel': NBEATSModel(input_chunk_length=24, output_chunk_length=12),
    'TCNModel': TCNModel(input_chunk_length=24, output_chunk_length=12),
    'TransformerModel': TransformerModel(input_chunk_length=24, output_chunk_length=12)
}

database_names = {
    'Cadastro Nacional de Obras(CNO)': 'cno',
    'Cadastro Nacional de Obras(CNO): Obras Pequenas 33% Primeiro Percentil': 'cno_small',
    'Cadastro Nacional de Obras(CNO): Obras Medias 33% Segundo Percentil': 'cno_medium',
    'Cadastro Nacional de Obras(CNO): Obras Grandes 33% Ultimo Percentil': 'cno_big',
    'Vagas de emprego(Sine e Indeed)': 'jobs',
    'Sites de noticias': 'news',
    'Cadastro Geral de Empregados e Desempregados (Caged): Admissoes - Desligamentos': 'caged_diff',
    'Cadastro Geral de Empregados e Desempregados (Caged): Saldo de Empregados': 'caged_balance',
    'Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD): Total': 'pnad_total',
    'Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD): Empregados Formais': 'pnad_formal',
    'Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD): Empregados Informais': 'pnad_informal',
}

database_short_names = {
    'CNO': 'Cadastro Nacional de Obras(CNO)',
    'CNO Pequeno': 'Cadastro Nacional de Obras(CNO): Obras Pequenas 33% Primeiro Percentil',
    'CNO Medio': 'Cadastro Nacional de Obras(CNO): Obras Medias 33% Segundo Percentil',
    'CNO Grande': 'Cadastro Nacional de Obras(CNO): Obras Grandes 33% Ultimo Percentil',
    'Vagas de emprego(Sine e Indeed)': 'Vagas de emprego(Sine e Indeed)',
    'Sites de noticias': 'Sites de noticias',
    'Caged: Saldo': 'Cadastro Geral de Empregados e Desempregados (Caged): Saldo de Empregados',
    'Caged: Admissoes - Desligamentos': 'Cadastro Geral de Empregados e Desempregados (Caged): Admissoes - Desligamentos',
    'PNAD Total': 'Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD): Total',
    'PNAD Formais': 'Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD): Empregados Formais',
    'PNAD Informais': 'Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD): Empregados Informais',
}


@st.cache
def get_caged_data():
    if os.path.isfile('data/caged_cache.csv'):
        return pd.read_csv('data/caged_cache.csv')
    fields = ['periodo', 'cbo', 'ocupacao']
    caged_df = pd.read_csv('data/caged.csv', usecols=fields)
    caged_df['cbo'] = caged_df['cbo'].astype(str)
    print(caged_df['cbo'].nunique())
    caged_df = caged_df[caged_df.groupby(col)[col].transform('count').ge(300)]
    print(caged_df['cbo'].nunique())
    # print(caged_df['cbo'].value_counts().to_string())
    caged_df = caged_df.rename(columns={'periodo': 'date'})
    caged_df['date'] = np.vectorize(standardize_date)(caged_df['date'], '%Y-%m-%d %H:%M:%S.000')
    caged_df = caged_df.sort_values(by='date')
    caged_df = caged_df[caged_df['date'].map(len) == 10]
    caged_df.sort_values(by='date')
    caged_df['date'] = pd.to_datetime(caged_df['date'])
    caged_df.to_csv('data/caged_cache.csv')
    return caged_df


def standardize_date(date_string, date_format, out_date_format='%Y-%m-%d'):
    try:
        date = datetime.strptime(date_string, date_format)
    except ValueError:
        date = parse(date_string)
    return date.strftime(out_date_format)


def test_multi_variate(series_list, model, model_name):
    scaled_series = []
    for series in series_list:
        scaler_series = Scaler()
        scaled_series.append(scaler_series.fit_transform(series))

    intervals = [(-37, -25), (-25, -13), (-13, -1)]
    mape_list = []
    for j in range(len(intervals)):
        train_series_list = []
        val_series_list = []
        for k in range(len(series_list)):
            train_series, val_series = scaled_series[k][:intervals[j][0]], scaled_series[k][
                                                                           intervals[j][0]:intervals[j][1]]
            train_series_list.append(train_series)
            val_series_list.append(val_series)

        try:
            model.fit(train_series_list, verbose=True)
        except TypeError:
            model.fit(train_series_list)

        predicted_series = scaled_series[0]
        pred = model.predict(n=12, series=train_series_list[0])
        mape_list.append(mape(predicted_series, pred))
        if j == 0:
            pred_sum = pred
        else:
            pred_sum = pred_sum.append(pred)

        df1 = predicted_series[(int(len(predicted_series) * 0.5)):].pd_dataframe()
        df2 = pred_sum.plot(label='forecast', lw=2).pd_dataframe()
        print(df1)
        print(df2)
        st.write('{} - MAPE = [{:.2f}%, {:.2f}%, {:.2f}%] {:.2f}%'.format(model_name,
                                                                          mape_list[0],
                                                                          mape_list[1],
                                                                          mape_list[2],
                                                                          np.mean(mape_list)))
        # df = pd.merge(df, df_pred, left_index=True, right_index=True, how='outer')
        # st.write("Previsao")
        # st.line_chart(df, use_container_width=False, width=800)



col = 'cbo'
caged_df = get_caged_data()
df = pd.read_csv('data/merged_total.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date', drop=False)

type_of_model = st.sidebar.radio('Selecione se o modelo vai usar 1 ou mais dados para a predição',
                                 ['Univariado', 'Multivariado'])

if type_of_model == 'Univariado':
    selected_data = st.sidebar.selectbox(label="Selecione a base a ser prevista", index=5,
                                         options=list(database_short_names))
    selected_data = [selected_data]
elif type_of_model == 'Multivariado':
    selected_data = st.sidebar.selectbox(label="Selecione a base a ser prevista", index=5,
                                         options=list(database_short_names))
    database_short_names_filtered = list(database_short_names)
    database_short_names_filtered.remove(selected_data)
    secondary_selected_data = st.sidebar.multiselect(label="Selecione a(s) base(s) que vao ajudar na previsão",
                                         options=list(database_short_names_filtered))
    secondary_selected_data.insert(0, selected_data)
    selected_data = secondary_selected_data

print(selected_data)
database_long_name = database_short_names.get(selected_data[0])
selected_data_column = database_names.get(database_long_name)

st.title('FIEA: Previsão de série temporal')
if selected_data_column.startswith('caged'):
    temp = database_long_name.split(':')
    header = temp[0]
    subheader = temp[1]
    st.header(header)
    st.subheader(subheader)
else:
    st.header(database_long_name)

st.write("Dados")

if selected_data_column.startswith('caged'):
    cbo_unique_list = sorted(caged_df['ocupacao'].unique())
    selected_cbo = st.sidebar.selectbox(label="Selecione ocupação CBO", index=0, options=['Todos'] + cbo_unique_list)
    if selected_cbo != 'Todos':
        cbo_df = caged_df.loc[(caged_df['ocupacao'] == selected_cbo)]
        cbo_df = cbo_df[['date']]
        cbo_df = cbo_df.groupby(['date'])['date'].count().reset_index(name="{}".format('count'))
        cbo_df = cbo_df.groupby(pd.Grouper(key='date', freq='1M')).sum()
        cbo_df['ocupacao'] = selected_cbo
        cbo_df.index = cbo_df.index.strftime('%Y-%m-01')
        cbo_df.index = pd.to_datetime(cbo_df.index)
        cbo_df = cbo_df[['count']]
        cbo_df = cbo_df.rename(columns={'count': 'caged_cbo'})
        df = pd.merge(df, cbo_df, left_index=True, right_index=True, how='outer')
        selected_data_column = 'caged_cbo'



if type_of_model == 'Univariado':
    model_algorithm = st.sidebar.selectbox("Selecione o algoritmo",
                                       ['Exponential Smoothing', 'Auto ARIMA', 'ARIMA', 'Prophet'],
                                       index=0)
elif type_of_model == 'Multivariado':
    model_algorithm = st.sidebar.selectbox("Selecione o algoritmo",
                                           ['RNNModel', 'BlockRNNModel', 'NBEATSModel', 'TCNModel', 'TransformerModel'],
                                           index=0)

forecast_horizon = st.sidebar.slider(label='Horizonte da previsão (meses)',
                                     min_value=1,
                                     max_value=48,
                                     value=12)

dataframe_percentage = st.sidebar.slider(label='Porcentagem da base original a ser mostrada na previsão',
                                         min_value=0,
                                         max_value=100,
                                         value=30,
                                         format='%d')

test_button = st.sidebar.button(label='Rodar Teste', on_click=None)


def train_multi_variate(series_list, model):
    scaled_series = []
    for series in series_list:
        scaler_series = Scaler()
        scaled_series.append(scaler_series.fit_transform(series))

    try:
        model.fit(scaled_series, verbose=True)
    except TypeError:
        model.fit(scaled_series)
    return model


if type_of_model == 'Univariado':
    df = df[[selected_data_column, 'date']]
    first_idx = df[selected_data_column].first_valid_index()
    last_idx = df[selected_data_column].last_valid_index()
    df = df.loc[first_idx:last_idx]
    df = df.fillna(method='ffill')
    df = df.reset_index(drop=True)
    print(df)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    series = TimeSeries.from_dataframe(df, value_cols=selected_data_column)

    model = model_dict[model_algorithm]
    model.fit(series)
    prediction = model.predict(forecast_horizon)
elif type_of_model == 'Multivariado':
    series_list = []
    first_idx_list = []
    last_idx_list = []
    df_list = []
    selected_data_column_list = []
    for data in selected_data:
        database_long_name = database_short_names.get(data)
        selected_data_column = database_names.get(database_long_name)
        selected_data_column_list.append(selected_data_column)
        dg = df[[selected_data_column, 'date']]
        df_list.append(dg)
        first_idx_list.append(dg[selected_data_column].first_valid_index())
        last_idx_list.append(dg[selected_data_column].last_valid_index())

    first_idx = max(first_idx_list)
    last_idx = min(last_idx_list)

    for i in range(len(df_list)):
        dg = df_list[i]
        dg = dg.loc[first_idx:last_idx]
        dg = dg.interpolate(method='linear', limit=3, limit_area='inside')
        dg = dg.reset_index(drop=True)
        dg['date'] = pd.to_datetime(dg['date'])
        dg = dg.set_index('date')
        if i == 0:
            df = dg
        print(selected_data_column_list[i])
        print(dg.head(2))
        print(dg.tail(2))
        print('----------------')
        series = TimeSeries.from_dataframe(dg, value_cols=selected_data_column_list[i])

        series_list.append(series)
    model = train_multi_variate(series_list, model_dict[model_algorithm])
    prediction = model.predict(forecast_horizon, series=series_list[0])

dataframe_percentage = dataframe_percentage / 100
print(df)
st.area_chart(df, use_container_width=False, width=800)
# st.write(prediction_arima)
original_plot = series[(int(len(series) * 0.8)):].plot()
forecast_plot = series[-1:].append(prediction).plot()

df_pred = series[-1:].append(prediction).pd_dataframe()
df_pred = df_pred.rename(columns={df_pred.columns[0]: 'pred'})

date = df.index[0]
periods = df.shape[0] + forecast_horizon + 1
date_index = pd.date_range(date, periods=periods, freq='H')
df = df.tail(int(len(df.index) * dataframe_percentage))
df = pd.merge(df, df_pred, left_index=True, right_index=True, how='outer')

st.write("Previsao")
st.line_chart(df, use_container_width=False, width=800)
