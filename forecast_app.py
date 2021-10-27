from darts import TimeSeries
from utils import model_dict, database_names, database_short_names
from utils import get_caged_data, test_multi_variate, train_multi_variate
import streamlit as st
import pandas as pd
import numpy as np
from tabulate import tabulate

# DATA LOADING
caged_df = get_caged_data()
df = pd.read_csv('data/merged_total.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date', drop=False)


# SIDEBAR BEGIN

# - SELECT MODEL TYPE
type_of_model = st.sidebar.radio('Selecione se o modelo vai usar 1 ou mais dados para a predição',
                                 ['Univariado', 'Multivariado'],
                                 index=1)

# - SELECT FORECAST DATA
if type_of_model == 'Univariado':
    selected_data = st.sidebar.selectbox(label="Selecione a base a ser prevista", index=6,
                                         options=list(database_short_names))
    selected_data = [selected_data]
elif type_of_model == 'Multivariado':
    selected_data = st.sidebar.selectbox(label="Selecione a base a ser prevista", index=6,
                                         options=list(database_short_names))
    database_short_names_filtered = list(database_short_names)
    database_short_names_filtered.remove(selected_data)

    # - SELECT FORECAST MULTIVARIATE AUXILIARY DATA
    auxiliary_selected_data = st.sidebar.multiselect(label="Selecione a(s) base(s) que vao ajudar na previsão",
                                                     options=list(database_short_names_filtered),
                                                     default=['PNAD Formais', 'PNAD Informais', 'CNO Grande'])
    auxiliary_selected_data.insert(0, selected_data)
    selected_data = auxiliary_selected_data


# - SELECT CBO IF FORECAST DATA IS CAGED
database_long_name = database_short_names.get(selected_data[0])
forecast_data_column = database_names.get(database_long_name)

if forecast_data_column.startswith('caged'):
    cbo_unique_list = sorted(caged_df['ocupacao'].unique())
    selected_cbo = st.sidebar.selectbox(label="Selecione ocupação CBO", index=0, options=['Todos'] + cbo_unique_list)
    if selected_cbo != 'Todos':
        # print(tabulate(caged_df.head(10), headers='keys', tablefmt='psql'))
        cbo_df = caged_df.loc[(caged_df['ocupacao'] == selected_cbo)]  # filter caged to selected cbo
        cbo_df = cbo_df[['date']]
        cbo_df = cbo_df.groupby(['date'])['date'].count().reset_index(name="{}".format('count'))  # transform df in quantity over time
        cbo_df.date = pd.to_datetime(cbo_df.date)
        cbo_df = cbo_df.groupby(pd.Grouper(key='date', freq='1M')).sum()  # transform df in quantity per month over time
        # print(tabulate(cbo_df.head(10), headers='keys', tablefmt='psql'))
        cbo_df['ocupacao'] = selected_cbo
        cbo_df.index = cbo_df.index.strftime('%Y-%m-01')
        cbo_df.index = pd.to_datetime(cbo_df.index)
        cbo_df = cbo_df[['count']]
        cbo_df = cbo_df.rename(columns={'count': 'caged_cbo'})
        df = pd.merge(df, cbo_df, left_index=True, right_index=True, how='outer')
        forecast_data_column = 'caged_cbo'


# - SELECT ALGORITHM
if type_of_model == 'Univariado':
    model_algorithm = st.sidebar.selectbox("Selecione o algoritmo",
                                           ['Exponential Smoothing', 'Auto ARIMA', 'ARIMA', 'Prophet'],
                                           index=0)
elif type_of_model == 'Multivariado':
    model_algorithm = st.sidebar.selectbox("Selecione o algoritmo",
                                           ['RNNModel', 'BlockRNNModel', 'NBEATSModel', 'TCNModel', 'TransformerModel'],
                                           index=1)


# - SELECT FORECAST WINDOW
forecast_horizon = st.sidebar.slider(label='Horizonte da previsão (meses)',
                                     min_value=1,
                                     max_value=48,
                                     value=12)

# - SELECT FORECAST ZOOM
dataframe_percentage = st.sidebar.slider(label='Zoom dos dados originais na previsão',
                                         min_value=0,
                                         max_value=100,
                                         value=30,
                                         format='%d')

# - RUN TEST
test_button = None
if type_of_model == 'Multivariado':
    test_button = st.sidebar.button(label='Rodar Teste')

# SIDEBAR END


# TRAIN
if type_of_model == 'Univariado':
    df = df[[forecast_data_column, 'date']]
    # get the inverval where there is data
    first_idx = df[forecast_data_column].first_valid_index()
    last_idx = df[forecast_data_column].last_valid_index()
    df = df.loc[first_idx:last_idx]

    df = df.fillna(method='ffill')
    df = df.reset_index(drop=True)
    # print(df)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    forecast_series = TimeSeries.from_dataframe(df, value_cols=forecast_data_column)
    series_list = [forecast_series]
    model = model_dict[model_algorithm]
    model.fit(forecast_series)
    prediction = model.predict(forecast_horizon)
elif type_of_model == 'Multivariado':
    series_list = []
    first_idx_list = []
    last_idx_list = []
    df_list = []
    selected_data_column_list = []
    for i in range(len(selected_data)):
        if i == 0:
            current_data_column = forecast_data_column
            selected_data_column_list.append(forecast_data_column)
        else:
            current_data_column = database_names.get(database_short_names.get(selected_data[i]))
            selected_data_column_list.append(current_data_column)

        dg = df[[current_data_column, 'date']]
        df_list.append(dg)
        first_idx_list.append(dg[current_data_column].first_valid_index())
        last_idx_list.append(dg[current_data_column].last_valid_index())

    # get the inverval where there is data for every auxiliary data
    first_idx = max(first_idx_list)
    last_idx = min(last_idx_list)

    for i in range(len(df_list)):
        dg = df_list[i]
        dg = dg.loc[first_idx:last_idx]
        # fill the gaps
        dg = dg.interpolate(method='linear', limit=3, limit_area='inside')
        dg = dg.reset_index(drop=True)
        dg['date'] = pd.to_datetime(dg['date'])
        dg = dg.set_index('date')
        if i == 0:
            df = dg
        series = TimeSeries.from_dataframe(dg, value_cols=selected_data_column_list[i])

        series_list.append(series)
    model, forecast_series = train_multi_variate(series_list, model_dict[model_algorithm])
    prediction = model.predict(forecast_horizon, series=forecast_series)

# PAGE

# - MAIN TITLE
st.title('FIEA: Previsão de série temporal')
if forecast_data_column.startswith('caged'):
    temp = database_long_name.split(':')
    header = temp[0]
    subheader = temp[1]
    st.header(header)
    st.subheader(subheader)
else:
    st.header(database_long_name)

# - DATA
st.write("Dados")
dataframe_percentage = dataframe_percentage / 100
st.area_chart(df, use_container_width=False, width=800)

index_query = (int(len(df.index) * dataframe_percentage)) * -1
df_data = forecast_series[index_query:].pd_dataframe()
df_data = df_data.rename(columns={df_data.columns[0]: forecast_data_column})

df_pred = forecast_series[-1:].append(prediction).pd_dataframe()
df_pred = df_pred.rename(columns={df_pred.columns[0]: 'pred'})

# print(tabulate(df_pred, headers='keys', tablefmt='psql'))

df_graph = pd.merge(df_data, df_pred, left_index=True, right_index=True, how='outer')
df_graph.columns = [str(col) for col in df_graph.columns]

# - FORECAST
st.write("Previsão")
st.line_chart(df_graph, use_container_width=False, width=800)

# - TEST RESULT CHART IF BUTTON PRESSED
if test_button:
    df_test_graph, mape_list = test_multi_variate(series_list, model_dict[model_algorithm], model_algorithm)
    mape_str = '{} - MAPE = [{:.2f}%, {:.2f}%, {:.2f}%] {:.2f}%'.format(model_algorithm,
                                                                        mape_list[0],
                                                                        mape_list[1],
                                                                        mape_list[2],
                                                                        np.mean(mape_list))
    st.write("Teste")
    st.write(mape_str)
    st.line_chart(df_test_graph, use_container_width=False, width=800)

# PAGE ENDED
