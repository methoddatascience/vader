import dash_html_components as html
import dash_core_components as dcc
from dash import Dash
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_table
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# intialize the dash app
app = Dash(__name__)
application = app.server

# load the dataset
df_data = pd.read_csv('data/red_crown_sample_data1.csv')

# remove date attributes
df_data = df_data.drop(['submission_date','first_contact','contract_length'], axis=1)

# convert bool attributes upgraded
df_data['upgraded'] = df_data.upgraded.apply(lambda x: str(x))
df_data['upgraded'] = df_data.upgraded.apply(lambda x: x.replace('False','0'))
df_data['upgraded'] = df_data.upgraded.apply(lambda x: x.replace('True','1'))

# convert upgraded data from object to number
df_data['upgraded'] = df_data.upgraded.apply(lambda x: int(x))

# convert categorical attributes gender
df_data['gender'] = df_data.gender.apply(lambda x: x.replace("Male","0"))
df_data['gender'] = df_data.gender.apply(lambda x: x.replace("Female","1"))

# convert gender data from object to number
df_data['gender'] = df_data.gender.apply(lambda x: int(x))

# convert bool attributes customer
df_data['customer'] = df_data.customer.apply(lambda x: str(x))
df_data['customer'] = df_data.customer.apply(lambda x: x.replace('False','0'))
df_data['customer'] = df_data.customer.apply(lambda x: x.replace('True','1'))

# convert customer data from object to number
df_data['customer'] = df_data.customer.apply(lambda x: int(x))

# select only numeric attributes
numeric_features = df_data[['age','upgraded','gender','category','score1','score2','score3','customer']]

# create the train dataset of known customers
df_customers = df_data[df_data['customer']==1]
df_non_customers = df_data[df_data['customer']==0]

# Drop the Id and customer values from the dataset
df_customers = df_customers.drop(['ID','customer'],axis=1)
df_non_customers = df_non_customers.drop(['ID','customer'],axis=1)

## Convert dataframe into list and then into a numpy array
train = df_customers.values.tolist()
train = np.array(train)

target_data = df_non_customers.values.tolist()
target_data = np.array(target_data)

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
pred = kmeans.fit_predict(train)

kmeans_target_data = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
kmeans_target_data.fit(train)
pred_target_data = kmeans_target_data.predict(target_data)

#save updated dataset
df_customers['pred_cluster'] = pred
df_non_customers['pred_cluster'] = pred_target_data

df_non_customers = df_non_customers[['pred_cluster','age','category','gender', 'upgraded','score1', 'score2', 'score3']]
df_customers = df_customers[['pred_cluster','age','category','gender', 'upgraded','score1', 'score2', 'score3']]

PAGE_SIZE = 5
dropdown_val = ['age', 'score1', 'score2', 'score3']
# define color palette
colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]
# define color palette
colors_1 = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]
colors_2 = ["#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]

#todo : mutlipage dash or have multiple modules and call from main
app.layout = html.Div([
						html.Div([
							html.H1("Red Crown Consulting"),
		                    html.P("Customer look-alike tool")
						],className="jumbotron text-center"),

    dcc.Tabs(id="tabs", children=[
		dcc.Tab(label='Existing Customers', children=[
			html.Div([
						html.H6("Clustering of known customers"),
						html.Hr()
						]),

			dcc.Dropdown(className="col-md-2", style={"margin-bottom": "10px"}, id="dropdown_x1",
						 options=[
							 {'label': val, 'value': val} for val in dropdown_val
						 ],
						 value=dropdown_val[0]
						 ),

			dcc.Dropdown(className="col-md-2", id="dropdown_y1",
						 options=[
							 {'label': val, 'value': val} for val in dropdown_val
						 ],
						 value=dropdown_val[1]
						 ),
			html.Div([
                    dcc.Graph(id="cluster_customers")
                    ],style={"padding": "20px"}),

            html.Div([
					html.Br()
					]),

			dash_table.DataTable(
				id='table-filtering_tab1',
				columns=[
					{"name": i, "id": i} for i in sorted(df_customers.columns)
				],

				pagination_settings={
					'current_page': 0,
					'page_size': PAGE_SIZE
				},
				pagination_mode='be',
				filtering='be',
				filtering_settings=''
			)


		]),



        dcc.Tab(label='Potential Customers', children=[
			html.Div([
						html.H6("Clustering of potential customers"),
						html.Hr()
						]),

                        dcc.Dropdown(className="col-md-2", style={"margin-bottom": "10px"}, id="dropdown_x2",
                        options=[
                        {'label': val, 'value': val} for val in dropdown_val
                        ],
                        value=dropdown_val[0]
                        ),

                    	dcc.Dropdown(className="col-md-2", id="dropdown_y2",
                    		options=[
                    		{'label': val, 'value': val} for val in dropdown_val
                    		],
                    		value=dropdown_val[1]
                    	),

                        html.Div([
                        		dcc.Graph(id="cluster_potential")
                        		],style={"padding": "20px"}),

                        html.Div([
						html.Br()
						]),


                        dash_table.DataTable(
                            id='table-filtering_tab2',
                            columns=[
                                {"name": i, "id": i} for i in sorted(df_non_customers.columns)
                            ],
                            pagination_settings={
                                'current_page': 0,
                                'page_size': PAGE_SIZE
                            },
                            pagination_mode='be',

                            filtering='be',
                            filtering_settings=''
                        )
        ]),

		dcc.Tab(label='Overlap of LookAlike Audience', children=[
			html.Div([
				html.H6("Overlap of two clustering plots"),
				html.Hr()
			]),

			dcc.Dropdown(className="col-md-2", style={"margin-bottom": "10px"}, id="dropdown_x3",
						 options=[
							 {'label': val, 'value': val} for val in dropdown_val
						 ],
						 value=dropdown_val[0]
						 ),

			dcc.Dropdown(className="col-md-2", id="dropdown_y3",
						 options=[
							 {'label': val, 'value': val} for val in dropdown_val
						 ],
						 value=dropdown_val[1]
						 ),

			html.Div([
				dcc.Graph(id="cluster_overlap")
			], style={"padding": "20px"}),

			html.Div([
				html.Br()
			]),
		]),
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
@app.callback(Output('cluster_customers', 'figure'), [Input('dropdown_x1', 'value'), Input('dropdown_y1', 'value')])
def update_graph(x_val, y_val):
	# count number of clusters
	num_of_clusters = df_customers["pred_cluster"].nunique()
	# create empty data list to store traces
	data = []
	# plot the actual labels
	for i in range(num_of_clusters):
		# split up the clusters to visualize
	    cluster_df = df_customers[df_customers["pred_cluster"] == i]
	    data.append({
	                "x": cluster_df[x_val],
	                "y": cluster_df[y_val],
	                "type": "scatter",
	                "name": "Cluster_"+ str(i),
	                "mode": "markers",
	                "marker": dict(
	                    color = colors[i],
	                    size = 13
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
	  "title": "Customer Dataset - "+ x_val + " vs " + y_val,
	  "xaxis": {
	    "domain": [0, 1],
	    "title": x_val
	  },
	  "yaxis": {
	    "domain": [0, 1],
	    "title": y_val
	  }
	}

	fig = {"data":data, "layout": layout}
	return fig

@app.callback(
    Output('table-filtering_tab1', 'data'),
    [Input('table-filtering_tab1', 'pagination_settings'),
     Input('table-filtering_tab1', 'filtering_settings')])
def update_graph(pagination_settings, filtering_settings):
    print(filtering_settings)
    filtering_expressions = filtering_settings.split(' && ')
    dff = df_customers
    for filter in filtering_expressions:
        if ' eq ' in filter:
            col_name = filter.split(' eq ')[0]
            filter_value = filter.split(' eq ')[1]
            dff = dff.loc[dff[col_name] == filter_value]
        if ' > ' in filter:
            col_name = filter.split(' > ')[0]
            filter_value = float(filter.split(' > ')[1])
            dff = dff.loc[dff[col_name] > filter_value]
        if ' < ' in filter:
            col_name = filter.split(' < ')[0]
            filter_value = float(filter.split(' < ')[1])
            dff = dff.loc[dff[col_name] < filter_value]
    return dff.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
    ].to_dict('rows')


@app.callback(
    Output('table-filtering_tab2', 'data'),
    [Input('table-filtering_tab2', 'pagination_settings'),
     Input('table-filtering_tab2', 'filtering_settings')])
def update_graph(pagination_settings, filtering_settings):
    print(filtering_settings)
    filtering_expressions = filtering_settings.split(' && ')
    dff = df_customers
    for filter in filtering_expressions:
        if ' eq ' in filter:
            col_name = filter.split(' eq ')[0]
            filter_value = filter.split(' eq ')[1]
            dff = dff.loc[dff[col_name] == filter_value]
        if ' > ' in filter:
            col_name = filter.split(' > ')[0]
            filter_value = float(filter.split(' > ')[1])
            dff = dff.loc[dff[col_name] > filter_value]
        if ' < ' in filter:
            col_name = filter.split(' < ')[0]
            filter_value = float(filter.split(' < ')[1])
            dff = dff.loc[dff[col_name] < filter_value]
    return dff.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
    ].to_dict('rows')

@app.callback(Output('cluster_potential', 'figure'), [Input('dropdown_x2', 'value'), Input('dropdown_y2', 'value')])
def update_graph(x_val, y_val):
	# count number of clusters
	num_of_clusters = df_non_customers["pred_cluster"].nunique()
	# create empty data list to store traces
	data = []
	# plot the actual labels
	for i in range(num_of_clusters):
	    # split up the clusters to visualize
	    cluster_df = df_non_customers[df_non_customers["pred_cluster"] == i]
	    data.append({
	                "x": cluster_df[x_val],
	                "y": cluster_df[y_val],
	                "type": "scatter",
	                "name": "Cluster_"+ str(i),
	                "mode": "markers",
	                "marker": dict(
	                    color = colors[i],
	                    size = 13
	                )
	            })

    # data.append({
    #             "x": [190134.88923284,116928.23926923,44090.25903961],
    #             "y": [32.57335128,35.01601022,37.36368854],
    #             "type": "scatter",
    #             "name": "Centroids",
    #             "mode": "markers",
    #             "marker": dict(
    #                 color = "#F781BF",
    #                 size = 13
    #             )
    #         })

	layout = {
	  "hovermode": "closest",
	  "margin": {
	    "r": 10,
	    "t": 25,
	    "b": 40,
	    "l": 60
	  },
	  "title": "Potential Customer Dataset - " + x_val + " vs "+y_val,
	  "xaxis": {
	    "domain": [0, 1],
	    "title": x_val
	  },
	  "yaxis": {
	    "domain": [0, 1],
	    "title": y_val
	  }
	}

	fig = {"data":data, "layout": layout}
	return fig


@app.callback(Output('cluster_overlap', 'figure'),
			  [ Input('dropdown_x3', 'value'), Input('dropdown_y3', 'value')])
def update_graph(x_val, y_val):

	# create empty data list to store traces
	trace1 = []

	# plot the actual labels
	num_of_clusters = df_customers["pred_cluster"].nunique()
	for i in range(3):
		# split up the clusters to visualize and extract sepal length and width
		cluster_df = df_customers[df_customers["pred_cluster"] == i]
		trace1 = go.Scattergl(
			x=cluster_df[x_val],
			y=cluster_df[y_val],
			# type='scatter',
			mode='markers',
			name='class_' + str(i),
			marker=dict(
				color=colors_1[i],
				size=18
				# shape='square' #this is still not available -- need to check documentation and update plotly version

			)
		)

		num_of_clusters2 = df_non_customers["pred_cluster"].nunique()
		trace2 = []
		for j in range(num_of_clusters2):
			# split up the clusters to visualize and extract sepal length and width
			cluster_df2 = df_non_customers[df_non_customers["pred_cluster"] == j]
			trace2 = go.Scattergl(
				x=cluster_df2[x_val],
				y=cluster_df2[y_val],
				# type='scatter',
				mode='markers',
				name='class_' + str(j),
				marker=dict(
					color=colors_2[j],
					size=8
					#shape='star'

				)
			)



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
		"title": "LookAlike Audience Dataset - " + x_val + " vs " + y_val,
		"xaxis": {
			"title": x_val
		},
		"yaxis": {
			"title": y_val
		}
	}

	data = [trace1, trace2]
	fig = {"data": data, "layout": layout}
	return fig


if __name__ == '__main__':
    app.run_server(debug=True)
