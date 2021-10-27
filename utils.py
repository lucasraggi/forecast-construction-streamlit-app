from darts.models import ExponentialSmoothing, AutoARIMA, ARIMA
from darts.models import NBEATSModel, RNNModel, TransformerModel, BlockRNNModel, TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from dateutil.parser import parse
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import os


n_epochs = 30
model_dict = {
    'Exponential Smoothing': ExponentialSmoothing(),
    'Auto ARIMA': AutoARIMA(),
    'ARIMA': ARIMA(),
    'RNNModel': RNNModel(input_chunk_length=24, n_epochs=n_epochs),
    'BlockRNNModel': BlockRNNModel(input_chunk_length=24, output_chunk_length=12, n_epochs=n_epochs),
    'NBEATSModel': NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=n_epochs),
    'TCNModel': TCNModel(input_chunk_length=24, output_chunk_length=12, n_epochs=n_epochs),
    'TransformerModel': TransformerModel(input_chunk_length=24, output_chunk_length=12, n_epochs=n_epochs)
}

database_names = {
    'Cadastro Nacional de Obras(CNO)': 'cno',
    'Cadastro Nacional de Obras(CNO): Obras Pequenas 33% Primeiro Percentil': 'cno_small',
    'Cadastro Nacional de Obras(CNO): Obras Medias 33% Segundo Percentil': 'cno_medium',
    'Cadastro Nacional de Obras(CNO): Obras Grandes 33% Ultimo Percentil': 'cno_big',
    'Vagas de emprego(Sine e Indeed)': 'jobs',
    'Sites de noticias': 'news',
    'Cadastro Geral de Empregados e Desempregados (Caged): Admissoes - Desligamentos': 'caged_diff',
    'Cadastro Geral de Empregados e Desempregados (Caged): Estoque de Empregados': 'caged_balance',
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
    'Caged: Estoque': 'Cadastro Geral de Empregados e Desempregados (Caged): Estoque de Empregados',
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
    caged_df = caged_df[caged_df.groupby('cbo')['cbo'].transform('count').ge(300)]
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


def test_multi_variate(series_list, model):
    scaled_series = []
    # normalize series betwen -1 and 1
    for series in series_list:
        scaler_series = Scaler()
        scaled_series.append(scaler_series.fit_transform(series))

    # intervals for [3 years ago to 2 years ago, 2 years ago to 1 year ago, 1 year ago to now]
    intervals = [(-37, -25), (-25, -13), (-13, -1)]
    mape_list = []
    for j in range(len(intervals)):
        train_series_list = []
        val_series_list = []
        for k in range(len(series_list)):
            train_series = scaled_series[k][:intervals[j][0]]
            val_series = scaled_series[k][intervals[j][0]:intervals[j][1]]
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

        # get the last half of the data so it can be easier to see
        df1 = predicted_series[(int(len(predicted_series) * 0.5)):].pd_dataframe()
        df1 = df1.rename(columns={df1.columns[0]: 'dado verdadeiro'})

        # print(tabulate(df1, headers='keys', tablefmt='psql'))

        df2 = pred_sum.pd_dataframe()
        df2 = df2.rename(columns={df2.columns[0]: 'previsão'})

        # print(tabulate(df2, headers='keys', tablefmt='psql'))

        df_test_graph = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
        df_test_graph.columns = [str(col) for col in df_test_graph.columns]

    return df_test_graph, mape_list


def train_multi_variate(series_list, model):
    scaled_series = []
    for series in series_list:
        series_scaler = Scaler()
        scaled_series.append(series_scaler.fit_transform(series))

    try:
        model.fit(scaled_series, verbose=True)
    except TypeError:
        model.fit(scaled_series)
    return model, scaled_series[0]
