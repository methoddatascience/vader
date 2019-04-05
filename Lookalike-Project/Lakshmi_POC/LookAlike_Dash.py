import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output
#from plotly.offline import plot, iplot # if you are running in Ipython notebook

import plotly.plotly as py
from plotly.graph_objs import *

from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

# intialize the dash app

app = Dash(__name__)

application = app.server

# load the input dataset
inputData = pd.read_csv('D:/MDS/Client/red_crown_sample_data1.csv')

# values for our dropdown
dropdown_val = ['score1', 'score2', 'score3']

# define color palette
colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]

###############################################################################
# build out the HTML

# begin making html page
app.layout = html.Div(className="container", style={"padding": "10px"}, children=[

	html.Div(className="jumbotron text-center", children=[
		html.H1("LookAlike Analysis"),
		html.P("Select the X and Y to visualize the data"),
		html.P ("Y can take 0/1 - None customer / Customer"),
		html.P("Use the slider to choose the number of clusters")

	]),

	dcc.Dropdown(className="col-md-4", style={"margin-bottom": "10px"}, id="dropdown_x",
				 options=[
					 {'label': val, 'value': val} for val in dropdown_val
				 ],
				 value=dropdown_val[0]
				 ),


	html.Br(),

	dcc.Slider(id="slider_n",
			   min=1,
			   max=9,
			   marks={i: '{}'.format(i) for i in range(1, 10)},
			   value=3,
			   ),

	html.Br(),

	html.Div(style={"padding": "20px"}, children=[
		dcc.Graph(id="cluster")
	])
])

# import external css
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"})

# import external javascript
app.scripts.append_script({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"})


##################################################
#                                                #
#       Callback Functions to update plots       #
#                                                #
##################################################
@app.callback(Output('cluster', 'figure'),
			  [ Input('slider_n', 'value')])
def update_graph(n):
	# build our dataframe

	inputData['customer'] = inputData['customer'].astype(int) # let it be False/True

	###############################################################################
	# prepare for modeling
	# assign your x and y values
	X = inputData[['score1', 'score2', 'score3']]

	#inputData[['score1', 'score2', 'score3']][0:1000].values

	# initialize kmeans algorithm
	kmeans = KMeans(n_clusters=n)

	# fit data to X
	kmeans.fit(X)

	# build our resutls dataframe
	inputData["predicted_classes"] = kmeans.labels_

	# count number of clusters
	num_of_clusters = inputData["predicted_classes"].nunique()
	# create empty data list to store traces
	data = []
	# plot the actual labels
	x_val = 'score1'
	y_val = 'customer'
	for i in range(num_of_clusters):
		# split up the clusters to visualize and extract sepal length and width
		cluster_df = inputData[inputData["predicted_classes"] == i]
		data.append({
			"x": cluster_df[x_val],
			"y": cluster_df[y_val],
			"type": "scatter",
			"name": f"class_{i}",
			"mode": "markers",
			"marker": dict(
				color=colors[i],
				size=10
			)
		})

	layout = {
		"hovermode": "closest",
		"margin": {
			"r": 10,
			"t": 25,
			"b": 40,
			"l": 60
		},
		"title": f"LookAlike Audience Dataset - {x_val} vs {y_val}",
		"xaxis": {
			"title": f"{x_val}"
		},
		"yaxis": {
			"title": f"{y_val}"
		}
	}

	fig = {"data": data, "layout": layout}
	return fig


if __name__ == '__main__':
	app.run_server()