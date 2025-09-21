import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

#prep data
#employment and realValAdd
realValAdd = pd.read_csv("spaceRealValueAdded.csv", )
employment = pd.read_csv('spaceEmployment.csv')

df_sis = pd.read_csv(r"yearly-number-of-objects-launched-into-outer-space.csv", )

df_sis_cleaned = df_sis[df_sis['Entity'] == "United States"]
df_sis_cleaned = df_sis_cleaned[["Year",  "Annual number of objects launched into outer space"]]
df_sis_cleaned = df_sis_cleaned[df_sis_cleaned["Year"].isin(range(2012, 2024))].reset_index(drop=True)

df_sis_cleaned_t = df_sis_cleaned.T
df_sis_cleaned_t.columns = df_sis_cleaned_t.iloc[0].astype(str)
df_sis_cleaned_t = df_sis_cleaned_t[1:]
df_sis_cleaned_t1 = df_sis_cleaned_t.reset_index().rename(columns={'index':' Economy Employment by Industry'})
df_sis_cleaned_t2 = df_sis_cleaned_t.reset_index().rename(columns={'index':'Real Value by Industry'})

df_finished_rvbi = pd.concat([realValAdd .dropna(subset='Real Value by Industry'),df_sis_cleaned_t2], ignore_index=True)
df_finished_eco = pd.concat([employment.dropna(subset=' Economy Employment by Industry'),df_sis_cleaned_t1], ignore_index=True)
employment.dropna(subset=' Economy Employment by Industry').tail(20)

#priv and gov
private = pd.read_csv('optimizedBusiness.csv')
private.rename(columns = {'Real Value Added by Industry': 'Real Value Added by Industry (millions)', 
                          'Real Gross Output by Industry': 'Real Gross Output by Industry (millions)',
                          'Employment ': 'Employment (thousands)',
                          'Compensation': 'Compensation (millions)'}, inplace= True)

df_sis = pd.read_csv(r"yearly-number-of-objects-launched-into-outer-space.csv", )
df_sis_cleaned = df_sis[df_sis['Entity'] == "United States"]
df_sis_cleaned = df_sis_cleaned[["Year",  "Annual number of objects launched into outer space"]]
numOb = df_sis_cleaned[df_sis_cleaned["Year"].isin(range(2012, 2024))].reset_index(drop=True)
numOb.head()
private['Annual number of objects launched into outer space'] = numOb['Annual number of objects launched into outer space']
private['Real Value Added by Industry (millions)'].iloc[0] = 79861
private['Real Value Added by Industry (millions)'] = private['Real Value Added by Industry (millions)'].astype(float)

gov = pd.read_csv('govSpends.csv')
gov.rename(columns = {'Real Value Added by Industry': 'Real Value Added by Industry (millions)', 
                          'Real Gross Output by Industry': 'Real Gross Output by Industry (millions)',
                          'Employment ': 'Employment (thousands)',
                          'Compensation': 'Compensation (millions)'}, inplace= True)
gov['Real Value Added by Industry (millions)'].iloc[0] = 22512
gov['Real Value Added by Industry (millions)'] = gov['Real Value Added by Industry (millions)'].astype(float)



#title and intro 
st.title('Liftoff in the Private Space Industry')
st.subheader('2025 CDC Project by Perry, Sarthak, and Amishi')
st.write('Business Track')

st.divider()

st.header('The Question')
st.write('As an investor, you may have noticed that space has become a big industry, and may be looking' 
          'to invest in it. In recent times, space stocks have grown rapidly. We can see the gross output ' \
          'of the space industry below. ')

total = pd.DataFrame()
total['Gross Output'] = private['Real Gross Output by Industry (millions)'] + gov['Real Gross Output by Industry (millions)']
total['Year'] = private['Year']
fig = px.line(total, x="Year", y="Gross Output", title='Gross Output of the Space Economy in Millions')
st.plotly_chart(fig, theme = 'streamlit')

st.write('Clearly it has grown a lot since in the last couple of years. We can see its break up outside of' \
' NASA and other government agencies as well. ')



'''
# Prepare data
x_vals = df_finished_rvbi.drop(columns=[df_finished_rvbi.columns[0]]).columns
heights = df_finished_rvbi.drop(columns=[df_finished_rvbi.columns[0]]).iloc[1]




# Create Plotly bar plot
fig = go.Figure(data=[
    go.Bar(
        x=x_vals,
        y=heights,
        marker_color='skyblue',
        marker_line_color='black'
        # width parameter removed
    )
])

fig.update_layout(
    title="Bar Plot with Plotly",
    xaxis_title="Year",
    yaxis_title="Value",
    bargap=0.2
)

st.plotly_chart(fig)
'''