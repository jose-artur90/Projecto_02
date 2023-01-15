import pandas as pd
import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Read and load data
df = pd.read_csv("hotel_booking.csv")
#Display data
df.head()

#Check if data holds duplicate values.
df.duplicated().any()

#Drop Duplicate entries 
df.drop_duplicates(inplace= False)
# will keep first row and others consider as duplicate


df.drop('company',axis=1, inplace = True)

print(df)

##
import missingno as msno
msno.bar(df)
plt.show()

msno.heatmap(df)

df.fillna(0, inplace = True)

df['is_canceled'].value_counts()

filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)

df[filter]

df = df[~filter]
print("******* df *******")
print(df[~filter])

#De onde vem a maioria dos hospedes
guest_city = df[df['is_canceled'] == 0]['country'].value_counts().reset_index()
guest_city.columns = ['Country', 'No of guests']
print("******* guest_city *******")
print(guest_city)

#Visualização de gráficos
import folium
from folium.plugins import HeatMap
import plotly.express as px

basemap = folium.Map()
guests_map = px.choropleth(guest_city, locations = guest_city['Country'],
                           color = guest_city['No of guests'], hover_name = guest_city['Country'])
guests_map.show()


#Preços dos quartos durante a noite para cada mês

data_resort = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]
data_city = df[(df['hotel'] == 'City Hotel') & (df['is_canceled'] == 0)]
resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
print("******* resort_hotel *******")
print(resort_hotel)
city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
print("******* city_hotel *******")
print(city_hotel)

final_hotel = pd.merge(resort_hotel, city_hotel, how="left", on="arrival_date_month")
final_hotel.rename(columns = {'adr_x':'price_for_resort_hotel'}, inplace = True)
final_hotel.rename(columns = {'adr_y':'price_for_city_hotel'}, inplace = True)
print(final_hotel)


import sort_dataframeby_monthorweek as sd

def sort_month(df, column_name):
    return sd.Sort_Dataframeby_Month(df, column_name)

final_Resort = sort_month(resort_hotel, 'arrival_date_month')
final_City = sort_month(city_hotel, 'arrival_date_month')
final_Hotel = sort_month(final_hotel, 'arrival_date_month')

print("******* Final Resort *******\n")
print(final_Resort)
print("******* Final City *******\n")
print(final_City)
print("******* Final Hotel *******\n")
print(final_Hotel)


plt.figure(figsize = (17, 8))
fig = px.line(final_Hotel, x = 'arrival_date_month', y = ['price_for_resort_hotel','price_for_city_hotel'],
        title = 'Room price per night over the Months', template = 'plotly_dark')
fig.show()


corr = df.corr()
#sns.<a onclick="parent.postMessage({'referent':'.seaborn.heatmap'}, '*')">heatmap(corr, annot = True, linewidths = 1)
#plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.show'}, '*')">show()

sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()

correlation = df.corr()['is_canceled'].abs().sort_values(ascending = False)
print("******* correlation *******")
print(correlation)

# dropping columns that are not useful

useless_col = ['days_in_waiting_list', 'arrival_date_year', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
               'reservation_status', 'country', 'days_in_waiting_list']

df.drop(useless_col, axis = 1, inplace = True)

#cat_cols = [<a onclick="parent.postMessage({'referent':'.kaggle.usercode.17158237.65503949.[4969,4972].col'}, '*')">col for <a onclick="parent.postMessage({'referent':'.kaggle.usercode.17158237.65503949.[4969,4972].col'}, '*')">col in df.columns if df[<a onclick="parent.postMessage({'referent':'.kaggle.usercode.17158237.65503949.[4969,4972].col'}, '*')">col].dtype == 'O']
cat_cols = [col for col in df.columns if df[col].dtype == 'O']
print("******* cat_cols *******")
print(cat_cols)
cat_df = df[cat_cols]
cat_df.head()

#cat_df['reservation_status_date'] = pd.<a onclick="parent.postMessage({'referent':'.pandas.to_datetime'}, '*')">to_datetime(cat_df['reservation_status_date'])
cat_df['reservation_status_date'] = pd.to_datetime(cat_df['reservation_status_date'])

cat_df['year'] = cat_df['reservation_status_date'].dt.year
cat_df['month'] = cat_df['reservation_status_date'].dt.month
cat_df['day'] = cat_df['reservation_status_date'].dt.day
cat_df.drop(['reservation_status_date','arrival_date_month'] , axis = 1, inplace = True)
cat_df.head()

# encoding categorical variables

cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})
cat_df['meal'] = cat_df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})
cat_df['market_segment'] = cat_df['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                                           'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})
cat_df['distribution_channel'] = cat_df['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,
                                                                       'GDS': 4})
cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                   'L': 7, 'B': 8})
cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})
cat_df['customer_type'] = cat_df['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})
cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})
cat_df.head()

num_df = df.drop(columns = cat_cols, axis = 1)
num_df.drop('is_canceled', axis = 1, inplace = True)
print("******* num_df *******")
print(num_df)

num_df.var()

#Colunas numéricas normalizadas que tem alta variância

num_df['lead_time'] = np.log(num_df['lead_time'] + 1)
num_df['arrival_date_week_number'] = np.log(num_df['arrival_date_week_number'] + 1)
num_df['arrival_date_day_of_month'] = np.log(num_df['arrival_date_day_of_month'] + 1)
num_df['agent'] = np.log(num_df['agent'] + 1)
num_df['adr'] = np.log(num_df['adr'] + 1)


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

#X = pd.<a onclick="parent.postMessage({'referent':'.pandas.concat'}, '*')">concat([cat_df, num_df], axis = 1)
X = pd.concat([cat_df, num_df], axis = 1)
y = df['is_canceled']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)

print(df)
print(X_train)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)


