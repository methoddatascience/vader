import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.plotly as py
from plotly.graph_objs import *
import dash_table
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# load the datasets
df_customers = pd.read_csv('../data/customers.csv')

df_customers = df_customers[['pred_cluster','age','category','gender', 'upgraded','score1', 'score2', 'score3']]

# values for our dropdown
# removed 'upgraded', 'gender', 'category' they grouped the dataset into 2 parts.
dropdown_val = ['age', 'score1', 'score2', 'score3']

# define color palette
colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]


app = dash.Dash()
###############################################################################
# build out the HTML

PAGE_SIZE = 5
# begin making html page
app.layout = html.Div([
						html.Div([
							html.H1("Red Crown Consulting"),
		                    html.P("Customer look-alike tool")
						],className="jumbotron text-center"),

						html.Div([
						html.H6("Clustering of known customers"),
						html.Hr()
						]),

                        dcc.Dropdown(className="col-md-2", style={"margin-bottom": "10px"}, id="dropdown_x",
                        options=[
                        {'label': val, 'value': val} for val in dropdown_val
                        ],
                        value=dropdown_val[0]
                        ),

                    	dcc.Dropdown(className="col-md-2", id="dropdown_y",
                    		options=[
                    		{'label': val, 'value': val} for val in dropdown_val
                    		],
                    		value=dropdown_val[1]
                    	),

                        html.Div([
                        		dcc.Graph(id="cluster")
                        		],style={"padding": "20px"}),

                        html.Div([
						html.Br()
						]),


                        dash_table.DataTable(
                            id='table-filtering',
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
					], className="container", style={"padding": "10px"})

# import external css
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"})
# import external javascript
app.scripts.append_script({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"})

##################################################
#                                                #
#       Callback Functions to update plots       #
#                                                #
##################################################
@app.callback(Output('cluster', 'figure'), [Input('dropdown_x', 'value'), Input('dropdown_y', 'value')])
def update_graph(x_val, y_val):
	# count number of clusters
	num_of_clusters = df_customers["pred_cluster"].nunique()
	# create empty data list to store traces
	data = []

    # data.append({
    #             "x": [190134.88923284,116928.23926923,44090.25903961],
    #             "y": [32.57335128,35.01601022,37.36368854],
    #             "type": "scatter",
    #             "name": "Centroids",
    #             "mode": "markers",
    #             "marker": dict(
    #                 color = "#F781BF",
    #                 size = 13)})

	# plot the actual labels
	for i in range(num_of_clusters):
	    # split up the clusters to visualize
	    cluster_df = df_customers[df_customers["pred_cluster"] == i]
	    data.append({
	                "x": cluster_df[x_val],
	                "y": cluster_df[y_val],
	                "type": "scatter",
	                "name": f"Cluster_{i}",
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
	  "title": f"Customer Dataset - {x_val} vs {y_val}",
	  "xaxis": {
	    "domain": [0, 1],
	    "title": f"{x_val}"
	  },
	  "yaxis": {
	    "domain": [0, 1],
	    "title": f"{y_val}"
	  }
	}

	fig = {"data":data, "layout": layout}
	return fig

@app.callback(
    Output('table-filtering', 'data'),
    [Input('table-filtering', 'pagination_settings'),
     Input('table-filtering', 'filtering_settings')])
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


if __name__ == '__main__':
    app.run_server()
