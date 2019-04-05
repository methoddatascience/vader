<<<<<<< HEAD
import sklearn
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np

from sklearn.cluster import KMeans
from sklearn import datasets

import pandas as pd


inputData = pd.read_csv('D:/MDS/Client/red_crown_sample_data1.csv')

#np.random.seed(5) # need to understand this

fig = tools.make_subplots(rows=2, cols=3,
						  print_grid=False,
						  specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
								 [{'is_3d': True, 'rowspan': 1}, None, None]])
scene = dict(
	camera=dict(
		up=dict(x=0, y=0, z=1),
		center=dict(x=0, y=0, z=0),
		eye=dict(x=2.5, y=0.1, z=0.1)
	),
	xaxis=dict(
		range=[-1, 4],
		title='Score 1',
		gridcolor='rgb(255, 255, 255)',
		zerolinecolor='rgb(255, 255, 255)',
		showbackground=True,
		backgroundcolor='rgb(230, 230,230)',
		showticklabels=False, ticks=''
	),
	yaxis=dict(
		range=[4, 8],
		title='Score 2',
		gridcolor='rgb(255, 255, 255)',
		zerolinecolor='rgb(255, 255, 255)',
		showbackground=True,
		backgroundcolor='rgb(230, 230,230)',
		showticklabels=False, ticks=''
	),
	zaxis=dict(
		range=[1, 8],
		title='Score 3',
		gridcolor='rgb(255, 255, 255)',
		zerolinecolor='rgb(255, 255, 255)',
		showbackground=True,
		backgroundcolor='rgb(230, 230,230)',
		showticklabels=False, ticks=''
	)
)

centers = [[1, 1], [-1, -1], [1, -1]]
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

X =  inputData[['score1' ,'score2' ,'score3']][0:1000].values
y = inputData['customer'][0:1000].astype(int)  # inputDat['customer']*1 alternative



estimators = {'k_means_lookAlike_3': KMeans(n_clusters=3)}
fignum = 1
for name, est in estimators.items():
	est.fit(X)
	labels = est.labels_

	trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
						 showlegend=False,
						 mode='markers',
						 marker=dict(
							 color=labels.astype(np.float),
							 line=dict(color='black', width=1)
						 ))
	fig.append_trace(trace, 1, fignum)

	fignum = fignum + 1

y = np.choose(y, [1, 0]).astype(np.float)

trace1 = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
					  showlegend=False,
					  mode='markers',
					  marker=dict(
						  color=y,
						  line=dict(color='black', width=1)))
fig.append_trace(trace1, 2, 1)

fig['layout'].update(height=900, width=900,
					 margin=dict(l=10, r=10))

fig['layout']['scene1'].update(scene)
fig['layout']['scene2'].update(scene)
fig['layout']['scene3'].update(scene)
fig['layout']['scene4'].update(scene)
#fig['layout']['scene5'].update(scene)


=======
import sklearn
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np

from sklearn.cluster import KMeans
from sklearn import datasets

import pandas as pd


inputData = pd.read_csv('D:/MDS/Client/red_crown_sample_data1.csv')

#np.random.seed(5) # need to understand this

fig = tools.make_subplots(rows=2, cols=3,
						  print_grid=False,
						  specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
								 [{'is_3d': True, 'rowspan': 1}, None, None]])
scene = dict(
	camera=dict(
		up=dict(x=0, y=0, z=1),
		center=dict(x=0, y=0, z=0),
		eye=dict(x=2.5, y=0.1, z=0.1)
	),
	xaxis=dict(
		range=[-1, 4],
		title='Score 1',
		gridcolor='rgb(255, 255, 255)',
		zerolinecolor='rgb(255, 255, 255)',
		showbackground=True,
		backgroundcolor='rgb(230, 230,230)',
		showticklabels=False, ticks=''
	),
	yaxis=dict(
		range=[4, 8],
		title='Score 2',
		gridcolor='rgb(255, 255, 255)',
		zerolinecolor='rgb(255, 255, 255)',
		showbackground=True,
		backgroundcolor='rgb(230, 230,230)',
		showticklabels=False, ticks=''
	),
	zaxis=dict(
		range=[1, 8],
		title='Score 3',
		gridcolor='rgb(255, 255, 255)',
		zerolinecolor='rgb(255, 255, 255)',
		showbackground=True,
		backgroundcolor='rgb(230, 230,230)',
		showticklabels=False, ticks=''
	)
)

centers = [[1, 1], [-1, -1], [1, -1]]
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

X =  inputData[['score1' ,'score2' ,'score3']][0:1000].values
y = inputData['customer'][0:1000].astype(int)  # inputDat['customer']*1 alternative



estimators = {'k_means_lookAlike_3': KMeans(n_clusters=3)}
fignum = 1
for name, est in estimators.items():
	est.fit(X)
	labels = est.labels_

	trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
						 showlegend=False,
						 mode='markers',
						 marker=dict(
							 color=labels.astype(np.float),
							 line=dict(color='black', width=1)
						 ))
	fig.append_trace(trace, 1, fignum)

	fignum = fignum + 1

y = np.choose(y, [1, 0]).astype(np.float)

trace1 = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
					  showlegend=False,
					  mode='markers',
					  marker=dict(
						  color=y,
						  line=dict(color='black', width=1)))
fig.append_trace(trace1, 2, 1)

fig['layout'].update(height=900, width=900,
					 margin=dict(l=10, r=10))

fig['layout']['scene1'].update(scene)
fig['layout']['scene2'].update(scene)
fig['layout']['scene3'].update(scene)
fig['layout']['scene4'].update(scene)
#fig['layout']['scene5'].update(scene)


>>>>>>> e10a8cb8be710a2f2bb9d38e3ab9f8ed3ccfb190
py.iplot(fig)