from ctypes import alignment
from inspect import stack
from json import tool
import logging
from math import radians
from multiprocessing.spawn import old_main_modules
from symtable import Symbol
from turtle import color
from dotenv import load_dotenv

from cv2 import sort
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

from googlesheets import GoogleSheets
import streamlit as st
import altair as alt
import requests
import json
import locale
import os

# st.set_page_config(layout="wide")
import base64


import pandas as pd
import numpy as np
import locale
locale.setlocale(locale.LC_MONETARY, 'en_IN')

import matplotlib.pyplot as plt

def configure():
    load_dotenv()

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "jpeg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def plot_line_charts(df, metric, breakup_select):
    try:
        c = df.iloc[:, df.columns.get_level_values(0)==metric]\
            .groupby(level=[0,(1 if breakup_select=='Loan' else 2)], axis=1)\
                .sum().reset_index()
        d = c.stack().reset_index()
        e = d[['level_0', breakup_select, metric]]\
            .merge(d[['level_0', 'Date']].dropna(), on='level_0')\
                .dropna(how='any').drop(columns='level_0')
        st.subheader('By {}'.format(breakup_select))
        chart = alt.Chart(e).mark_line()\
                .encode(x=alt.X('Date', axis=alt.Axis(format='%b-%Y')), 
                y=alt.Y(metric), color=breakup_select, tooltip=alt.Tooltip(metric, format=',.2f')).interactive()
        st.altair_chart(chart, use_container_width=True)
        st.subheader('Cumulative')
        # chart = alt.Chart(e.groupby('Date')[metric].sum().reset_index())\
        #         .mark_line().encode(x='Date', y=metric, tooltip=metric).interactive()
        chart = alt.Chart(e.groupby('Date')[metric].sum().reset_index())\
                .mark_line().encode(x=alt.X('Date', axis=alt.Axis(format='%b-%Y')), 
                y=alt.Y(metric), tooltip=alt.Tooltip(metric, format=',.2f')).interactive()
        st.altair_chart(chart, use_container_width=True)
    except:
        pass

def plot_bar_charts(df, metric, breakup_select):
    try:
        c = df.iloc[:, df.columns.get_level_values(0)==metric]\
            .groupby(level=[0,(1 if breakup_select=='Loan' else 2)], axis=1)\
                .sum().reset_index()
        d = c.stack().reset_index()
        e = d[['level_0', breakup_select, metric]]\
            .merge(d[['level_0', 'Date']].dropna(), on='level_0')\
                .dropna(how='any').drop(columns='level_0')
        st.subheader('By {}'.format(breakup_select))
        chart = alt.Chart(e).mark_bar()\
                .encode(x=alt.X('Date', axis=alt.Axis(format='%b-%Y'), bin=False), 
                y=alt.Y(metric, aggregate='sum', type='quantitative', stack=True),
                color=breakup_select, 
                tooltip=alt.Tooltip(metric, format=',.2f')).interactive()
        st.altair_chart(chart, use_container_width=True)
    except:
        pass

def plot_pie_charts(df, metric, breakup_select):
    try:
        c = df.iloc[:, df.columns.get_level_values(0)==metric]\
        .groupby(level=[0,(1 if breakup_select=='Loan' else 2)], axis=1)\
            .sum().reset_index()
        d = c.stack().reset_index()
        e = d[['level_0', breakup_select, metric]]\
        .merge(d[['level_0', 'Date']].dropna(), on='level_0')\
            .dropna(how='any').drop(columns='level_0')
        date_max = e['Date'].max()
        f = e.loc[e['Date']==date_max].drop(columns=['Date']).reset_index(drop=True)
        f['HalfAngle'] = f[metric].cumsum()
        f['HalfAngle'] = f['HalfAngle'] - f[metric]/2
        chart = alt.Chart(f).mark_arc().encode(
        theta=alt.Theta(field=metric, type="quantitative"),
        color=alt.Color(field=breakup_select, type="nominal"),
        tooltip=alt.Tooltip(field=breakup_select, type="nominal"),
        )
        text = alt.Chart(f).mark_text(radius=100).encode(
            text = alt.Text(metric, format=",.2f"),
            theta = alt.Theta(field='HalfAngle', type="quantitative"),
        )

        st.altair_chart(chart+text)
    except:
        pass

def get_gs_data():
    gc = GoogleSheets()
    configure()

    sheet_id = os.getenv('sheet_id')
    range_name = 'Data!B:F'

    values = gc.get_sheet_data(sheet_id, range_name)

    df = pd.DataFrame(values)
    df.columns = df.iloc[0,:].values
    df = df.iloc[1:,:]
    for x in ['Debit', 'Credit']:
        df[x] = pd.to_numeric(df[x].str.replace(',',''))
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    a = df.loc[df['Debit']>100000]
    a.rename(columns={'Debit':'Disbursement'}, inplace=True)
    a['Disbursement'].fillna(0, inplace=True)
    df = df.join(a['Disbursement'])
    a = df.groupby(['Date', 'Account','Loan'])\
        .agg({
            'Debit': np.sum, 
            'Credit': np.sum,
            'Disbursement': np.sum
            }).reset_index()
    return a

def main():
    # set_bg_hack('background.jpeg')
    
    a = get_gs_data()
    a.to_csv('data.csv', index=False)

    a = pd.read_csv('data.csv')

    a['Date'] = pd.to_datetime(a['Date'], format='%Y-%m-%d')
    date_min = a['Date'].min()
    date_max = a['Date'].max()
    dates = pd.DataFrame(pd.date_range(date_min, date_max, freq='D'))
    dates.columns = ['Date']
    b = a.pivot(index=['Date'], columns=['Loan', 'Account'], 
    values=['Debit', 'Credit', 'Disbursement'])
    b = b.reset_index()

    b.iloc[:, b.columns.get_level_values(0)=='Debit'] = \
        b.iloc[:, b.columns.get_level_values(0)=='Debit'].fillna(0).cumsum()
    b.iloc[:, b.columns.get_level_values(0)=='Credit'] = \
        b.iloc[:, b.columns.get_level_values(0)=='Credit'].fillna(0).cumsum()
    b.iloc[:, b.columns.get_level_values(0)=='Disbursement'] = \
        b.iloc[:, b.columns.get_level_values(0)=='Disbursement'].fillna(0).cumsum()

    num = b.iloc[:, b.columns.get_level_values(0)=='Credit'].values\
        - b.iloc[:, b.columns.get_level_values(0)=='Debit'].values\
        + b.iloc[:, b.columns.get_level_values(0)=='Disbursement'].values
    num = pd.DataFrame(num)
    old_idx = b.iloc[:, b.columns.get_level_values(0)=='Credit'].columns.to_frame()
    old_idx.rename(index={'Credit': 'Principal Repaid'})
    old_idx.iloc[:,0] = 'Principal Repaid'
    num.columns = pd.MultiIndex.from_frame(old_idx)
    b = pd.concat([b, num], axis=1)

    num = b.iloc[:, b.columns.get_level_values(0)=='Debit'].values\
        - b.iloc[:, b.columns.get_level_values(0)=='Credit'].values
    num = pd.DataFrame(num)
    old_idx = b.iloc[:, b.columns.get_level_values(0)=='Credit'].columns.to_frame()
    old_idx.rename(index={'Credit': 'Pending Amount'})
    old_idx.iloc[:,0] = 'Pending Amount'
    num.columns = pd.MultiIndex.from_frame(old_idx)
    b = pd.concat([b, num], axis=1)

    num = b.iloc[:, b.columns.get_level_values(0)=='Credit'].values\
        - b.iloc[:, b.columns.get_level_values(0)=='Principal Repaid'].values
    num = pd.DataFrame(num)
    old_idx = b.iloc[:, b.columns.get_level_values(0)=='Credit'].columns.to_frame()
    old_idx.rename(index={'Credit': 'Interest Paid'})
    old_idx.iloc[:,0] = 'Interest Paid'
    num.columns = pd.MultiIndex.from_frame(old_idx)
    b = pd.concat([b, num], axis=1)

    old_idx = b.columns.to_frame()
    b = dates.merge(b, on='Date', how='left')
    b = b.fillna(method='ffill')
    b.columns = pd.MultiIndex.from_frame(old_idx)

    b.to_csv('data2.csv', index=False)

    b.set_index('Date', inplace=True)

    # with requests.get('https://raw.githubusercontent.com/d3/d3-format/master/locale/en-IN.json') as c:
    #     in_format = json.loads(c.content)
    # alt.renderers.set_embed_options(formatLocale=in_format)

    st.title('Loans for Prestige Tranquil')

    page = st.selectbox('Choose your page', ['Current Snapshot', 
    'Total to be Paid', 'Pending Amount', 'Total Paid', 'Interest Paid',
    'Principal Repaid', 'Total Disbursement']) 
    
    add_breakup_select = st.sidebar.radio(
    'View breakup by',
    ['Loan', 'Account'])

    add_loan_multiselect = st.sidebar.multiselect(
    'Select loans',
    a.Loan.unique(),
    default=a.Loan.unique())

    add_account_multiselect = st.sidebar.multiselect(
    'Select accounts',
    a.loc[a['Loan'].isin(add_loan_multiselect)].Account.unique(),
    default=a.loc[a['Loan'].isin(add_loan_multiselect)].Account.unique())

    b = b.iloc[:,b.columns.get_level_values(2).isin(add_account_multiselect)]

    # if len(add_loan_multiselect) < 2:
    #     add_breakup_select = 'Account'


    personal_pending_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Pending Amount')
    &(b.columns.get_level_values(1)=='Personal Loan')].sum()
    personal_principal_repaid_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Principal Repaid')
    &(b.columns.get_level_values(1)=='Personal Loan')].sum()
    personal_interest_paid_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Interest Paid')
    &(b.columns.get_level_values(1)=='Personal Loan')].sum()
    personal_total_debit_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Debit')
    &(b.columns.get_level_values(1)=='Personal Loan')].sum()
    personal_total_disbursement_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Disbursement')
    &(b.columns.get_level_values(1)=='Personal Loan')].sum()
    personal_total_credit_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Credit')
    &(b.columns.get_level_values(1)=='Personal Loan')].sum()

    home_pending_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Pending Amount')
    &(b.columns.get_level_values(1)=='Home Loan')].sum()
    home_principal_repaid_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Principal Repaid')
    &(b.columns.get_level_values(1)=='Home Loan')].sum()
    home_interest_paid_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Interest Paid')
    &(b.columns.get_level_values(1)=='Home Loan')].sum()
    home_total_debit_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Debit')
    &(b.columns.get_level_values(1)=='Home Loan')].sum()
    home_total_disbursement_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Disbursement')
    &(b.columns.get_level_values(1)=='Home Loan')].sum()
    home_total_credit_amount = b.iloc[-1, (b.columns.get_level_values(0)=='Credit')
    &(b.columns.get_level_values(1)=='Home Loan')].sum()

    pending_amount = b.iloc[-1, b.columns.get_level_values(0)=='Pending Amount'].sum()
    principal_repaid_amount = b.iloc[-1, b.columns.get_level_values(0)=='Principal Repaid'].sum()
    interest_paid_amount = b.iloc[-1, b.columns.get_level_values(0)=='Interest Paid'].sum()
    total_debit_amount = b.iloc[-1, b.columns.get_level_values(0)=='Debit'].sum()
    total_disbursement_amount = b.iloc[-1, b.columns.get_level_values(0)=='Disbursement'].sum()
    total_credit_amount = b.iloc[-1, b.columns.get_level_values(0)=='Credit'].sum()

    if page == 'Current Snapshot':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader('Home Loan')
            # st.metric('Total to be paid', locale.currency(home_total_debit_amount, grouping=True), delta=None, delta_color="normal")
            st.metric('Total to be paid', str(round(home_total_debit_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({} x total disbursement)'.format(round(home_total_debit_amount/home_total_disbursement_amount, 2)))
            st.subheader('-')
            st.metric('Pending amount', str(round(home_pending_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total to be paid)'.format(round(home_pending_amount/home_total_debit_amount*100, 2)))
            st.subheader('=')
            st.metric('Total paid', str(round(home_total_credit_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total to be paid)'.format(round(home_total_credit_amount/home_total_debit_amount*100, 2)))
            st.subheader('-')
            st.metric('Interest paid', str(round(home_interest_paid_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total paid)'.format(round(home_interest_paid_amount/home_total_credit_amount*100, 2)))
            st.subheader('=')
            st.metric('Principal repaid', str(round(home_principal_repaid_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total disbursement)'.format(round(home_principal_repaid_amount/home_total_disbursement_amount*100, 2)))
            st.write('out of')
            st.metric('Total disbursement', str(round(home_total_disbursement_amount/10**5,1))+'L', delta=None, delta_color="normal")
        with col2:
            st.subheader('Personal Loan')
            st.metric('Total to be paid', str(round(personal_total_debit_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({} x total disbursement)'.format(round(personal_total_debit_amount/personal_total_disbursement_amount, 2)))
            st.subheader('-')
            st.metric('Pending amount', str(round(personal_pending_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total to be paid)'.format(round(personal_pending_amount/personal_total_debit_amount*100, 2)))
            st.subheader('=')
            st.metric('Total paid', str(round(personal_total_credit_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total to be paid)'.format(round(personal_total_credit_amount/personal_total_debit_amount*100, 2)))
            st.subheader('-')
            st.metric('Interest paid', str(round(personal_interest_paid_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total paid)'.format(round(personal_interest_paid_amount/personal_total_credit_amount*100, 2)))
            st.subheader('=')
            st.metric('Principal repaid', str(round(personal_principal_repaid_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total disbursement)'.format(round(personal_principal_repaid_amount/personal_total_disbursement_amount*100, 2)))
            st.write('out of')
            st.metric('Total disbursement', str(round(personal_total_disbursement_amount/10**5,1))+'L', delta=None, delta_color="normal")
        with col3:
            st.subheader('Total')
            st.metric('Total to be paid', str(round(total_debit_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({} x total disbursement)'.format(round(total_debit_amount/total_disbursement_amount, 2)))
            st.subheader('-')
            st.metric('Pending amount', str(round(pending_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total to be paid)'.format(round(pending_amount/total_debit_amount*100, 2)))
            st.subheader('=')
            st.metric('Total paid', str(round(total_credit_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total to be paid)'.format(round(total_credit_amount/total_debit_amount*100, 2)))
            st.subheader('-')
            st.metric('Interest paid', str(round(interest_paid_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total paid)'.format(round(interest_paid_amount/total_credit_amount*100, 2)))
            st.subheader('=')
            st.metric('Principal repaid', str(round(principal_repaid_amount/10**5,1))+'L', delta=None, delta_color="normal")
            st.write('({}% of total disbursement)'.format(round(principal_repaid_amount/total_disbursement_amount*100, 2)))
            st.write('out of')
            st.metric('Total disbursement', str(round(total_disbursement_amount/10**5,1))+'L', delta=None, delta_color="normal")       


    if page == 'Total to be Paid':
        st.metric('Total to be paid', locale.currency(total_debit_amount, grouping=True), delta=None, delta_color="normal")
        plot_pie_charts(b, 'Debit', add_breakup_select)
        plot_line_charts(b, 'Debit', add_breakup_select)
        c = b.iloc[:, b.columns.get_level_values(0)=='Debit'].diff().groupby(pd.Grouper(freq='W-MON')).sum()
        plot_bar_charts(c, 'Debit', add_breakup_select)
    if page == 'Pending Amount':
        st.metric('Pending amount', locale.currency(pending_amount, grouping=True), delta=None, delta_color="normal")
        plot_pie_charts(b, 'Pending Amount', add_breakup_select)
        plot_line_charts(b, 'Pending Amount', add_breakup_select)
    if page == 'Total Paid':
        st.metric('Total paid', locale.currency(total_credit_amount, grouping=True), delta=None, delta_color="normal")
        plot_pie_charts(b, 'Credit', add_breakup_select)
        plot_line_charts(b, 'Credit', add_breakup_select)
        c = b.iloc[:, b.columns.get_level_values(0)=='Credit'].diff().groupby(pd.Grouper(freq='W-MON')).sum()
        plot_bar_charts(c, 'Credit', add_breakup_select)
    if page == 'Principal Repaid':
        st.metric('Principal repaid', locale.currency(principal_repaid_amount, grouping=True), delta=None, delta_color="normal")
        plot_pie_charts(b, 'Principal Repaid', add_breakup_select)
        plot_line_charts(b, 'Principal Repaid', add_breakup_select)
        c = b.iloc[:, b.columns.get_level_values(0)=='Principal Repaid'].diff().groupby(pd.Grouper(freq='W-MON')).sum()
        plot_bar_charts(c, 'Principal Repaid', add_breakup_select)
    if page == 'Interest Paid':
        st.metric('Interest paid', locale.currency(interest_paid_amount, grouping=True), delta=None, delta_color="normal")
        plot_pie_charts(b, 'Interest Paid', add_breakup_select)
        plot_line_charts(b, 'Interest Paid', add_breakup_select)
        c = b.iloc[:, b.columns.get_level_values(0)=='Interest Paid'].diff().groupby(pd.Grouper(freq='W-MON')).sum()
        plot_bar_charts(c, 'Interest Paid', add_breakup_select)
    if page == 'Total Disbursement':
        st.metric('Total disbursement', locale.currency(total_disbursement_amount, grouping=True), delta=None, delta_color="normal")
        plot_pie_charts(b, 'Disbursement', add_breakup_select)
        plot_line_charts(b, 'Disbursement', add_breakup_select)
        c = b.iloc[:, b.columns.get_level_values(0)=='Disbursement'].diff().groupby(pd.Grouper(freq='W-MON')).sum()
        plot_bar_charts(c, 'Disbursement', add_breakup_select)


if __name__ == '__main__':
    main()