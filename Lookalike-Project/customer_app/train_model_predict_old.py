import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# load the dataset
df_data = pd.read_csv('../data/red_crown_sample_data1.csv')

# remove date attributes
df_data = df_data.drop(['submission_date','first_contact','contract_length'],axis=1)

# convert bool attributes upgraded
df_data['upgraded'] = df_data.upgraded.apply(lambda x: str(x))
df_data['upgraded'] = df_data.upgraded.apply(lambda x: x.replace('False','0'))
df_data['upgraded'] = df_data.upgraded.apply(lambda x: x.replace('True','1'))

# convert upgraded data fron object to number
df_data['upgraded'] = df_data.upgraded.apply(lambda x: int(x))

# convert categorical attributes gender
df_data['gender'] = df_data.gender.apply(lambda x: x.replace("Male","0"))
df_data['gender'] = df_data.gender.apply(lambda x: x.replace("Female","1"))

# convert gender data fron object to number
df_data['gender'] = df_data.gender.apply(lambda x: int(x))

# convert bool attributes customer
df_data['customer'] = df_data.customer.apply(lambda x: str(x))
df_data['customer'] = df_data.customer.apply(lambda x: x.replace('False','0'))
df_data['customer'] = df_data.customer.apply(lambda x: x.replace('True','1'))

# convert customer data fron object to number
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

df_customers.to_csv('../data/customers.csv',index=False)
df_non_customers.to_csv('../data/non_customers.csv',index=False)
