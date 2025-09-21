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

branches_df = pd.read_csv('branches_df.csv')


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

graph1 = pd.DataFrame()
graph1['Year'] = private['Year']
graph1['priv'] = private['Real Gross Output by Industry (millions)']
graph1['gov'] = gov['Real Gross Output by Industry (millions)']
graph1['privL'] = graph1['priv'] - private['Annual number of objects launched into outer space']
graph1['privU']= graph1['priv'] + private['Annual number of objects launched into outer space']
graph1['govL'] = graph1['gov'] - private['Annual number of objects launched into outer space']
graph1['govU']= graph1['gov'] + private['Annual number of objects launched into outer space']
yearRev = graph1['Year'][::-1]

x = list(graph1['Year'])
x_rev = x[::-1]
y_upper = list(graph1['govU'])
y_lower = list(graph1['govL'])[::-1]
y_upper1 = list(graph1['privU'])
y_lower1 = list(graph1['privL'])[::-1]

fig1 = go.Figure()

# --- Government Plot ---
fig_gov = go.Figure()

fig_gov.add_trace(go.Scatter(
    x=x + x_rev,
    y=y_upper + y_lower,
    fill='toself',
    fillcolor='rgba(133,203,51,0.2)',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
    name='Government Range',
))
fig_gov.add_trace(go.Scatter(
    x=graph1['Year'],
    y=graph1['gov'],
    line_color='#85cb33',
    name='Government Gross Output',
    showlegend=True,
))

fig_gov.update_layout(
    title="Government Gross Output Range",
    xaxis_title="Year",
    yaxis_title="Gross Output (millions)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# --- Private Plot ---
fig_priv = go.Figure()

fig_priv.add_trace(go.Scatter(
    x=x + x_rev,
    y=y_upper1 + y_lower1,
    fill='toself',
    fillcolor='rgba(165,203,195,0.2)',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
    name='Private Range',
))
fig_priv.add_trace(go.Scatter(
    x=graph1['Year'],
    y=graph1['priv'],
    line_color='#a5cbc3',
    name='Private Gross Output',
    showlegend=True,
))

fig_priv.update_layout(
    title="Private Gross Output Range",
    xaxis_title="Year",
    yaxis_title="Gross Output (millions)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_gov, theme='streamlit', use_container_width=True)
    st.write('The area fill around the lines show the number of objects launched into space that year. As can be seen, the' \
    ' government gross output is very strongly correlated with the number of objects launched into space, especially after 2019, ' \
    'with a correlation value of 09436.')
with col2:
    st.plotly_chart(fig_priv, theme='streamlit', use_container_width=True)
    st.write('The private gross output are weakly correlated with a value of -0.2639. This goes against what we ')



fig2 = go.Figure()

num = st.slider("Pick a number of samples. Larger samples slow down the site.", 0, 9999)
sample = branches_df.groupby("Simulation").sample(num)

for sim, group in sample:
    fig2.add_trace(go.Scatter(
        x=group["Year"],
        y=group["Gross_Output"],
        mode='lines',
        line=dict(color='rgba(165,203,195,0.02)', width=1),
        showlegend=False
    ))

mean_by_year = branches_df.groupby("Year")["Gross_Output"].mean()
fig2.add_trace(go.Scatter(
    x=mean_by_year.index,
    y=mean_by_year.values,
    mode='lines',
    line=dict(color='#85cb33', width=2.5),
    name="Mean across simulations"
))

fig2.update_layout(
    title="Monte Carlo Simulations",
    xaxis_title="Year",
    yaxis_title="Gross Output",
    template="plotly_white"
)

st.plotly_chart(fig2, theme='streamlit')

