#SVD, PCA, Overlay two cluster graphs
#todo : add option to upload input data csv file
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.tools as tools
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
colors_1 = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]
colors_2 = ["#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]
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

	dcc.Dropdown(className="col-md-4", id="dropdown_y",
				 options=[
					 {'label': val, 'value': val} for val in dropdown_val
				 ],
				 value=dropdown_val[1]
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
			  [ Input('dropdown_x', 'value'), Input('dropdown_y', 'value'), Input('slider_n', 'value')])
def update_graph(x_val, y_val, n):
	# build our dataframe

	# build our dataframe


	df_customers = inputData[inputData['customer'] == 1]
	df_noncustomers = inputData[inputData['customer'] == 0]
	#df_customers['target'] = inputData['customer'].astype(int).values

	###############################################################################
	# prepare for modeling
	# assign your x and y values
	X = df_customers[['score1', 'score2', 'score3']]
	X2 = df_noncustomers[['score1', 'score2', 'score3']]

	#inputData[['score1', 'score2', 'score3']][0:1000].values

	# initialize kmeans algorithm
	kmeans = KMeans(n_clusters=n)


	# fit data to X
	kmeans.fit(X)


	# build our resutls dataframe
	df_customers["predicted_classes"] = kmeans.labels_


	# count number of clusters
	num_of_clusters = df_customers["predicted_classes"].nunique()

	# create empty data list to store traces
	trace1 = []

	# plot the actual labels

	for i in range(num_of_clusters):
		# split up the clusters to visualize and extract sepal length and width
		cluster_df = df_customers[df_customers["predicted_classes"] == i]
		trace1 = go.Scattergl(
			x=cluster_df[x_val],
			y=cluster_df[y_val],
			# type='scatter',
			mode='markers',
			name=f'class_{i}',
			marker=dict(
				color=colors_1[i],
				size=8
				# shape='square' #this is still not available -- need to check documentation and update plotly version

			)
		)
		# trace1.append({
		# 	"x": cluster_df[x_val],
		# 	"y": cluster_df[y_val],
		# 	"type": "scatter",
		# 	"name": f"class_{i}",
		# 	"mode": "markers",
		# 	"marker": dict(
		# 		color=colors[i],
		# 		size=8
		# 	)
		# })

		kmeans2 = KMeans(n_clusters=n)
		kmeans2.fit(X2)
		df_noncustomers["predicted_classes"] = kmeans2.labels_
		num_of_clusters2 = df_noncustomers["predicted_classes"].nunique()
		trace2 = []
		for j in range(num_of_clusters2):
			# split up the clusters to visualize and extract sepal length and width
			cluster_df2 = df_noncustomers[df_noncustomers["predicted_classes"] == j]
			trace2 = go.Scattergl(
				x=cluster_df2[x_val],
				y=cluster_df2[y_val],
				# type='scatter',
				mode='markers',
				name=f'class_{j}',
				marker=dict(
					color=colors_2[j],
					size=18
					#shape='star'

				)
			)
			# trace2.append({
			# 	"x": cluster_df2[x_val],
			# 	"y": cluster_df2[y_val],
			# 	"type": "scatter",
			# 	"name": f"class_{j}",
			# 	"mode": "markers",
			# 	"marker": dict(
			# 		color=colors[i],
			# 		size=10,
			# 		symbol='star'
			# 	)
			# })




	layout = {
		# "barmode":"overlay",
		"scattermode" : "overlay",
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
	# The below two lines of code does not give the required result
	data = [trace1, trace2]
	fig = {"data": data, "layout": layout}

	# fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.009, horizontal_spacing=0.009)
	# fig['layout']['margin'] = {'l': 60, 'r': 10, 'b': 40, 't': 25}
	#
	# fig.append_trace(trace1)
	# fig.append_trace(trace2)

	return fig


if __name__ == '__main__':
	app.run_server(debug=True)