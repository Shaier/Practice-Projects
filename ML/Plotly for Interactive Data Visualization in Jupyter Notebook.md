# Using Plotly Library for Interactive Data Visualization in Jupyter Notebook
```python
#Libraries
import pandas as pd
import numpy as np
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

#connect the JavaScript to our notebook (Plotly plots are interactive hence they use JS)
"the scripts that we are going to run will be executed in the Jupyter notebook. To connect Jupyter notebook with JavaScript"

init_notebook_mode(connected=True)

#import Cufflink and use it offline
cf.go_offline()

#Importing the data library+data
import seaborn as sns
dataset=sns.load_dataset('tips')
dataset.head()

dataset2=dataset[["total_bill", "tip", "size"]]
dataset2.iplot()

#Bar Plot
dataset.iplot(kind='bar', x=['time', 'sex'],y='total_bill')
dataset.mean().iplot(kind='bar') #bar chart
dataset.mean().iplot(kind='barh')#horizontal bar plots

#Scatter Plot
dataset.iplot(kind='scatter', x='total_bill', y='tip', mode='markers')

#Box Plot
dataset2.iplot(kind='box')

#Histogram
dataset['total_bill'].iplot(kind='hist',bins=25)

#Scatter Matrix Plot
dataset2.scatter_matrix()  #set of all of the scatter plots of the numerical columns

#Spread Plot
dataset[['total_bill','tip']].iplot(kind='spread') #shows the spread between two+ of the numerical columns at any particular point-- ex. total bill vs tip

#3D Plots
dataset2 = dataset[["total_bill", "tip", "size"]]
data = dataset2.iplot(kind='surface', colorscale='rdylbu')

####################################################################################################

#Geographical Plots

#Libraries
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import pandas as pd

#Creating a Data Dictionary
"dictionary that actually contains the data that you want to display on the map"
map_data = dict(type='choropleth',
            locations=['MI', 'CO', 'FL', 'IN'],
            locationmode='USA-states',
            colorscale='Portland',
            text=['Michigan', 'Colorado', 'Florida', 'Indiana'], #text contains a list of strings that will be displayed when the mouse hovers over the state location
            z=[1.0,2.0,3.0,4.0], # numerical values that will be displayed when the mouse hovers over the state location
            colorbar=dict(title="USA States") #colorbar is a dictionary. title key-> you can specify the text that will be displayed on the color bar.
           )

#Creating a Layout
map_layout = dict(geo = {'scope':'usa'}) #layout dictionary for the US

#Creating Graph Object
map_actual = go.Figure(data=[map_data], layout=map_layout)

#Plot
iplot(map_actual)

#Geographical Maps for the United States Using CSV

#Libraries and data
path='C:\\Users\\sagi\\Desktop\\bea-gdp-by-state.csv'
df=pd.read_csv(path)

#Adding abbreviation for the states
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

df['abbrev'] = df['Area'].map(us_state_abbrev)

#Creating a Data Dictionary

map_data = dict(type='choropleth',
            locations=df['abbrev'], #the geographical plot for all the states in the US will be printed
            locationmode='USA-states',
            colorscale='Reds',
            text=df['Area'],
            marker=dict(line=dict(color='rgb(255,0,0)', width=2)),
            z=df['2017'], #GDP for the year 2017
            colorbar=dict(title="GDP Per Capita - 2017")
           )

#Layout
map_layout = dict(title='USA States GDP Per Capita - 2017',
              geo=dict(scope='usa',
                         showlakes=True, #lakes will be displayed on the map with colors as specified by the RGB
                         lakecolor='rgb(85,173,240)')
             )
#Creating the graph object and passing it the data and layout dictionaries
map_actual = go.Figure(data=[map_data], layout=map_layout)

#Plotting
iplot(map_actual)

######################################################################3

#Geographical Maps for the World
path='C:\\Users\\sagi\\Desktop\\bea-gdp-by-state.csv'
df = pd.read_csv(path)
df.head()

#Data Dictionary
map_data = dict(
        type='choropleth',
        locations=df['Country Code'],
        z=df['2016'],
        text=df['Country'],
        colorbar={'title': 'World Population 2016'},
      )

#Layout dictionary
map_layout = dict(
    title='World Population 2016',
    geo=dict(showframe=False)
)


map_actual = go.Figure(data=[map_data], layout=map_layout)
iplot(map_actual)
```
# Resources
https://stackabuse.com/using-plotly-library-for-interactive-data-visualization-in-python/
