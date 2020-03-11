# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(path)
#data["Rating"].hist()
#plt.hist(data["Rating"])
filter1=data["Rating"]<=5
data=data[filter1]
data["Rating"].hist()
#Code starts here


#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
percent_null=(total_null/data.isnull().count())
missing_data=pd.concat([total_null, percent_null], keys=['Total','Percent'],axis=1)
print(missing_data)
data.dropna(inplace=True)
total_null_1=data.isnull().sum()
percent_null_1=(total_null_1/data.isnull().count())
print(total_null_1)
print(percent_null_1)
missing_data_1=pd.concat([total_null_1, percent_null_1], keys=['Total','Percent'],axis=1)
print(missing_data_1)
# code ends here


# --------------

#Code starts here
import seaborn as sns
sns.set(style="ticks")
#data = sns.load_dataset("exercise")
g = sns.catplot(x="Category", y="Rating", data=data, kind="box")


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#print(data["Installs"].value_counts())
data['Installs'] = data['Installs'].map(lambda x: x.rstrip(',+')).str.replace(",","")
data['Installs'] =data['Installs'].astype("int")
#print(data['Installs'])

le=LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
#print(data['Installs'])
#Code starts here
sns.regplot(x="Installs", y="Rating", data=data)



#Code ends here



# --------------
#Code starts here
#print(data["Price"].value_counts())
data["Price"]=data["Price"].map(lambda x: x.lstrip("$"))
data["Price"]=data["Price"].astype('float')
#print(data["Price"])
sns.regplot(x="Price", y="Rating", data=data)
#Code ends here


# --------------

#Code starts here

#Finding the length of unique genres
print( len(data['Genres'].unique()) , "genres")

#Splitting the column to include only the first genre of each app
data['Genres'] = data['Genres'].str.split(';').str[0]

#Grouping Genres and Rating
gr_mean=data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()

print(gr_mean.describe())

#Sorting the grouped dataframe by Rating
gr_mean=gr_mean.sort_values('Rating')

print(gr_mean.head(1))

print(gr_mean.tail(1))

#Code ends here



# --------------

#Code starts here
#print(data["Last Updated"])
data["Last Updated"]=pd.to_datetime(data["Last Updated"])
max_date=max(data["Last Updated"])
Diff_dates= max_date-data["Last Updated"]
data["Last Updated Days"]=Diff_dates.dt.days
sns.regplot(x="Last Updated Days", y="Rating", data=data)
#print(data.head())
#Code ends here


