import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

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
govBranch = pd.read_csv('govBranch.csv')


#title and intro 
st.title('Liftoff in the Space Industry')
st.subheader('2025 CDC Project by Perry, Sarthak, and Amishi')
st.write('An investigation into potential growth and investment opportunities in the private space industry.    ')

st.divider()

st.header('Our Question')
st.write('As an investor, you may have noticed that space has become a big industry, and may be looking' 
          'to invest in it. In recent times, space stocks have grown rapidly. We can see the gross output ' \
          'of the space industry below. ')

total = pd.DataFrame()
total['Gross Output'] = private['Real Gross Output by Industry (millions)'] + gov['Real Gross Output by Industry (millions)']
total['Year'] = private['Year']
fig = px.line(total, x="Year", y="Gross Output", title='Gross Output of the Space Economy in Millions')
st.plotly_chart(fig, theme = 'streamlit')

st.write('Clearly it has grown quite a bit. In our search to explain this, we came accross a dataset that showed the number'
' of objects launched into space annually, which can be shown below.  ')

fig1 = px.line(private, x="Year", y="Annual number of objects launched into outer space", 
              title='Annual Number of Objects Launched into Outer Space')
fig1.add_vline(
    x=2019,
    line_width=2,
    line_dash="dash",
    line_color="#85cb33",
    annotation_text="2019",
    annotation_position="top"
)
st.plotly_chart(fig1, theme = 'streamlit')

st.write("We saw the number of objects launched 'lifted off' at about 2019, growing, seemingly exponentially. Naturally, " \
"with the growth of private space companies, we wanted to see whether it had to do with private players or the government. " \
"Below are the graphs of the gross output of the private and government space industries, with a fill area showing the" \
" number of objects launched in an given year. ")

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
    st.write(' As can be seen, the' \
    ' government gross output is very strongly correlated with the number of objects launched into space, especially after 2019, ' \
    'with a correlation value of 09436.')
with col2:
    st.plotly_chart(fig_priv, theme='streamlit', use_container_width=True)
    st.write('The private gross output are weakly correlated with a value of -0.2639. This goes against what we initially thought,' \
    'especially since, looking at news in that era, we saw a few big headlines regarding space.')

st.write('With this information, we turned to the news for more information on what could be causing that sudden increase in ' \
'number of objects launched. We found two big things: the beginning of launches for the Starlink, as well as the creation of the' \
' United States Space Force, which could have caused a surge in launches. However, with a lack of data it is hard to say for sure.')

st.write('As an investor, you want to know the future of the industry - how it will grow how big. The last released estimate ' \
' on the size of the space industry was in 2023, and it was estimated to be $202 billion. We bring you, many models used to estimate' \
' the size of the space industry.')

st.divider()

st.header('Monte Carlo Simulation')

st.subheader('Private Gross Output')

fig2 = go.Figure()

num = st.slider("Pick a number of samples. Larger samples slow down the site.", 10, 9999, 100, key = 'priv' )

# Get unique simulation IDs and sample them
sim_ids = branches_df['Simulation'].unique()
sampled_ids = np.random.choice(sim_ids, size=min(num, len(sim_ids)), replace=False)

for sim in sampled_ids:
    group = branches_df[branches_df['Simulation'] == sim]
    fig2.add_trace(go.Scatter(
        x=group["Year"],
        y=group["Gross_Output"],
        mode='lines',
        line=dict(color='rgba(165,203,195,0.1)', width=1),
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
    title="Monte Carlo Simulations for Private Gross Output ",
    xaxis_title="Year",
    yaxis_title="Gross Output",
    template="plotly_white"
)

st.plotly_chart(fig2, theme='streamlit')

st.write('We can see that the average tends to taper off to one point. If we just look at that average line:')

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=mean_by_year.index,
    y=mean_by_year.values,
    mode='lines',
    line=dict(color='#85cb33', width=2.5),
    name="Mean across simulations"
))

fig3.update_layout(
    title="Monte Carlo Simulations",
    xaxis_title="Year",
    yaxis_title="Gross Output",
    template="plotly_white"
)

col3,col4 = st.columns(2)
with col3:
    st.plotly_chart(fig3, theme='streamlit')
with col4:
    st.write(
    'The model predicts an increase in growth rate of the space industry till 2024, and then a bit of decrease. '
    'This may indicate that the industry would have reached its peak and then will stabilize. However, there are other '
    'approaches we can take to understand how the space industry will grow'
    )

st.subheader('Government Gross Output')


fig4 = go.Figure()

numb = st.slider("Pick a number of samples. Larger samples slow down the site.", 10, 9999, 100, key = 'gov')

# Get unique simulation IDs and sample them
sims = govBranch['Simulation'].unique()
sampled = np.random.choice(sim_ids, size=min(num, len(sims)), replace=False)

for sim in sampled:
    group = govBranch[govBranch['Simulation'] == sim]
    fig4.add_trace(go.Scatter(
        x=group["Year"],
        y=group["Gross_Output"],
        mode='lines',
        line=dict(color='rgba(165,203,195,0.1)', width=1),
        showlegend=False
    ))

mean_by_year = govBranch.groupby("Year")["Gross_Output"].mean()
fig4.add_trace(go.Scatter(
    x=mean_by_year.index,
    y=mean_by_year.values,
    mode='lines',
    line=dict(color='#85cb33', width=2.5),
    name="Mean across simulations"
))

fig4.update_layout(
    title="Monte Carlo Simulations for Private Gross Output ",
    xaxis_title="Year",
    yaxis_title="Gross Output",
    template="plotly_white"
)

st.plotly_chart(fig4, theme='streamlit')

st.write('Yet again, we see that the line tapers off to a equillibrium')

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=mean_by_year.index,
    y=mean_by_year.values,
    mode='lines',
    line=dict(color='#85cb33', width=2.5),
    name="Mean across simulations"
))

fig5.update_layout(
    title="Monte Carlo Simulations for Government Gross Output",
    xaxis_title="Year",
    yaxis_title="Gross Output",
    template="plotly_white"
)

col5,col6 = st.columns(2)
with col5:
    st.plotly_chart(fig5, theme='streamlit')
with col6:
    st.write(
    'The model predicts an increase in the growth rate of the government space industry. Unlike ' \
    'the private space indsutry, which tends to stop growing after 2025, the government space industry is ' \
    'projected to continue growing even after that, which may be due to the settlement of the Space Force.'
    )

st.divider()

st.header('ARIMA Model')

st.subheader('Government Gross Output')

st.write('We can use an ARIMA model to see what it projects is the growth of the ')

model = ARIMA(gov['Real Gross Output by Industry (millions)'], order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=3)


years_hist = gov['Year']
output_hist = gov['Real Gross Output by Industry (millions)']

n_forecast = 3
years_forecast = np.arange(years_hist.iloc[-1] + 1, years_hist.iloc[-1] + 1 + n_forecast)
output_forecast = forecast.values



fig7 = go.Figure()

fig7.add_trace(go.Scatter(
    x=years_hist,
    y=output_hist,
    mode='lines+markers',
    name='Historical',
    line=dict(color='#a5cbc3')
))

fig7.add_trace(go.Scatter(
    x=years_forecast,
    y=output_forecast,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#85cb33')
))

fig7.update_layout(
    title="ARIMA Forecast for Government Gross Output",
    xaxis_title="Year",
    yaxis_title="Gross Output (millions)",
    template="plotly_white"
)

st.plotly_chart(fig7, theme='streamlit')

st.write('We see that this model, likely due to the limited data, ' \
'projects limited growth in the next few years for gross output. We can do the same for public gross output.')

st.subheader('Private Gross Output')

model2 = ARIMA(gov['Real Gross Output by Industry (millions)'], order=(1,1,1))
model_fit2 = model2.fit()
forecast2 = model_fit2.forecast(steps=3)


years_hist2 = private['Year']
output_hist2 = private['Real Gross Output by Industry (millions)']

n_forecast = 3
years_forecast2 = np.arange(years_hist2.iloc[-1] + 1, years_hist2.iloc[-1] + 1 + n_forecast)
output_forecast2 = forecast2.values

fig8 = go.Figure()

fig8.add_trace(go.Scatter(
    x=years_hist2,
    y=output_hist2,
    mode='lines+markers',
    name='Historical',
    line=dict(color='#a5cbc3')
))

fig8.add_trace(go.Scatter(
    x=years_forecast2,
    y=output_forecast2,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#85cb33')
))

fig8.update_layout(
    title="ARIMA Forecast for Private Gross Output",
    xaxis_title="Year",
    yaxis_title="Gross Output (millions)",
    template="plotly_white"
)

st.plotly_chart(fig8, theme='streamlit')