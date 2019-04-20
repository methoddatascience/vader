import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# load the dataset
df_data = pd.read_csv('../data/red_crown_sample_data1.csv')

# remove date attributes
df_data = df_data.drop(['submission_date','first_contact','contract_length'], axis=1)

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

# create the dataset of known customers
df_customers = df_data[df_data['customer']==1]

# create the dataset of non customers
df_non_customers = df_data[df_data['customer']==0]

# Drop the Id and customer values from the dataset
df_customers = df_customers.drop(['ID','customer'],axis=1)
df_non_customers = df_non_customers.drop(['ID','customer'],axis=1)

# Convert dataframe into list and then into a numpy array
customer_list = df_customers.values.tolist()
customer_list = np.array(customer_list)

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
pred_labels = kmeans.fit_predict(customer_list)

#save updated customer dataset
df_customers['pred_cluster'] = pred_labels
df_customers.to_csv('../data/customers.csv',index=False)

# create train and validation datasets from clustered dataset
X = np.array(df_customers[['age','upgraded','gender','category','score1','score2','score3']].values.tolist())
y = np.ravel(df_customers[['pred_cluster']].values.tolist())
#Note: reason for using ravel was due to the warning message i got below:
#DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().

# validation options and evaluation metric
num_folds = 10
seed = 42
validation_size = 0.30
scoring = 'accuracy'

# train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# Make predictions on validation dataset using the best model
svm = SVC()
svm.fit(X_train, y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))

# Get values to make predictions on new dataset
X = df_non_customers[['age','upgraded','gender','category','score1','score2','score3']].values
predictions_new = svm.predict(X)

# Add the predicted cluster label values to the dataset
df_non_customers['pred_cluster'] = predictions_new

# non_customers class distribution by pred_cluster
print(df_non_customers.groupby('pred_cluster').size())

df_non_customers.to_csv('../data/non_customers.csv',index=False)
