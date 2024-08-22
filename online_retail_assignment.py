import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('OnlineRetail.csv',encoding='unicode_escape')

df.info()

print(df.isnull().sum())

x=df['Quantity']>0
unit_price=df['UnitPrice']>0
df=df[x]
df=df[unit_price]
print(df)

df['Total_Sales']=df['UnitPrice']*df['Quantity']

print(df.describe())

df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
df.info()

# for i in df.columns:
#     if ((df[i].dtype=='int64') or (df[i].dtype=='float64')):
#         plt.boxplot(df[i])
#         plt.xlabel(i)
#         plt.show()
#         print(i)

outliers_columns=['Total_Sales','UnitPrice','Quantity']


for x in outliers_columns:

    Q1 = df[x].quantile(0.25)  
    Q3 = df[x].quantile(0.75)
    IQR = Q3 - Q1
    LF = Q1 - 1.5 * IQR
    UF = Q3 + 1.5 * IQR
    df = df[(df[x] >= LF) & (df[x] <= UF)]


# for i in df.columns:
#     if ((df[i].dtype=='int64') or (df[i].dtype=='float64')):
#         plt.boxplot(df[i])
#         plt.xlabel(i)
#         plt.show()
#         print(i)

df=df.drop_duplicates()

df.dropna(inplace=True)
#dropping the null values


df.info()

# print(df)        

#TOP 10 CUSTOMERS SALES WISE

print(df)
# customers_spending=df.groupby(df['CustomerID'])['Total_Sales'].sum().reset_index()
# customers_spending=customers_spending.sort_values(by='Total_Sales',ascending=False)
# print(customers_spending)

# sns.barplot(x=customers_spending.iloc[0:11,0],y=customers_spending.iloc[0:11,1])
# plt.title('Top 10 Customers Sales Wise')
# ax.bar_label(ax.containers[0])
# plt.show()

# TOP COUNTRIES WITH RESPECT TO CUSTOMERS

# country_customer=df.groupby(df['Country'])['CustomerID'].nunique().reset_index().rename(columns={'CustomerID':'Total_Customer_country_Wise'})
# country_customer=country_customer.sort_values(by='Total_Customer_country_Wise',ascending=False)
# print(country_customer)

# ax=sns.barplot(country_customer,x=country_customer.iloc[0:10,0],y=country_customer.iloc[0:10,1],errorbar=None)
# ax.bar_label(ax.containers[0])
# plt.title('Top 10 Countries With Respect To Customers')
# plt.show()

# TOP 10 COUNTRIES SALES WISE

# country_sales=df.groupby(df['Country'])['Total_Sales'].sum().reset_index().rename(columns={'Total_Sales':'Total_Sales_Country_Wise'})
# country_sales=country_sales.sort_values(by='Total_Sales_Country_Wise',ascending=False)
# print(country_sales)

# ax=sns.barplot(country_sales,x=country_sales.iloc[0:10,0],y=country_sales.iloc[0:10,1],errorbar=None)
# ax.bar_label(ax.containers[0])
# plt.title('Top 10 Countries With Respect To Sales')
# plt.show()

df['Date'] = pd.to_datetime(df['InvoiceDate'],dayfirst=True)
df['Year'] = df.Date.dt.year
df['Month'] = df.Date.dt.month
print(df)

#MONTH WISE SALES

# month_wise_sales=df.groupby('Month')['Total_Sales'].sum().reset_index().rename(columns={'Total_Sales':'Month_Wise_Sales'})
# month_wise_sales=month_wise_sales.sort_values(by='Month')


# def month_name(month):
#     if month==1:
#         return 'January'
#     if month==2:
#         return 'February'
#     if month==3:
#         return 'March'
#     if month==4:
#         return 'April'
#     if month==5:
#         return 'May'
#     if month==6:
#         return 'June'
#     if month==7:
#         return 'July'
#     if month==8:
#         return 'August'
#     if month==9:
#         return 'September'
#     if month==10:
#         return 'October'
#     if month==11:
#         return 'November'
#     else:
#         return 'December'
      
  
# month_wise_sales['Month Name']=month_wise_sales['Month'].apply(month_name)   

# print(month_wise_sales)    
    
# ax=sns.barplot(month_wise_sales,x=month_wise_sales.iloc[:,2],y=month_wise_sales.iloc[:,1],errorbar=None)
# ax.bar_label(ax.containers[0])
# plt.title('Month Wise Sales')
# plt.show()

#YEAR WISE SALES

# year_wise_sales=df.groupby('Year')['Total_Sales'].sum().reset_index().rename(columns={'Tatal_Sales':'Year_Wise_Sales'})
# print(year_wise_sales)

# ax=sns.barplot(x=year_wise_sales.iloc[:,0],y=year_wise_sales.iloc[:,1])
# ax.bar_label(ax.containers[0])
# plt.title('Year Wise Sales')
# plt.show()

#TOP 20 MOST SOLD ITEMS
# most_sole_items=df.groupby('Description')['Quantity'].count().reset_index().rename(columns={'Quantity':'Total_Sold'})
# most_sole_items=most_sole_items.sort_values(by='Total_Sold',ascending=False)
# print(most_sole_items)

# ax=sns.barplot(x=most_sole_items.iloc[0:20,0],y=most_sole_items.iloc[0:20,1])
# ax.bar_label(ax.containers[0])
# plt.xticks(rotation=90)
# plt.title('Year Wise Sales')
# plt.show()

#SEGMENTING THE CUSTOMERS ON THE BASIS OF RFM ANALYSIS

# import datetime as dt
# today_date = dt.datetime(2011, 12, 1)

final = df['InvoiceDate'].max()+ pd.DateOffset(1)
df['recency'] = final - df['InvoiceDate']
recency_df = df.groupby(df['CustomerID']).min()['recency'].reset_index()
recency_df['recency']=recency_df['recency'].dt.days
# print(recency_df)

frequecy_df=df.groupby('CustomerID')['InvoiceNo'].count().reset_index().rename(columns={'InvoiceNo':'Frequency'})
# print(frequecy_df)

monetary_df=df.groupby('CustomerID')['Total_Sales'].sum().reset_index().rename(columns={'Total_Sales':'Monetary'})
# print(monetary_df)

frequecy_df=monetary_df.merge(frequecy_df,on='CustomerID')
rfm=recency_df.merge(frequecy_df,on='CustomerID')
rfm.set_index('CustomerID', inplace=True)
# rfm=rfm.reset_index()
rfm.columns=['recency','Monetary','Frequency']
print(rfm)

from sklearn.preprocessing import StandardScaler


sc=StandardScaler()
rfm_scaled=sc.fit_transform(rfm)
df1=pd.DataFrame(rfm_scaled,columns=['recency','Monetary','Frequency'])
print(df1)


from  sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss=[]

for k in range(1,20):
    KM=KMeans(n_clusters=k,init='k-means++')
    KM.fit(df1)
    wcss.append(KM.inertia_)
    


plt.plot(np.arange(1,20),wcss)
plt.scatter(np.arange(1,20),wcss,color='red')
plt.show()

model=KMeans(n_clusters=5,init='k-means++')
cluster_id=model.fit_predict(df1)
rfm['clusterid']=cluster_id

print(rfm)

cluster_purchase=rfm.groupby('clusterid')['Monetary'].sum()
cluster_recency=rfm.groupby('clusterid')['recency'].sum()
cluster_frequency=rfm.groupby('clusterid')['Frequency'].sum()

# print()

plt.subplot(3,1,1)
sns.barplot(x=[1,2,3,4,5],y=cluster_purchase)
plt.subplot(3,1,2)
sns.barplot(x=[1,2,3,4,5],y=cluster_recency)
plt.subplot(3,1,3)
sns.barplot(x=[1,2,3,4,5],y=cluster_frequency)
plt.show()

#inference 

#CLUSTER 1=AFTER VISUALIZING THE BAR PLOT OF ALL CLUSTERS WE CAN SAY THAT CUSTOMERS FROM CLUSTER 1 ARE FREQUENT HIGH SPENDERS 
# AND WE CAN CONSIDER THEM AS A RETAINED CUSTOMER BUT THEY HAVE NOT DONE 
# ANY TRANSACTION IN RECENT PERIOD. SO I ADVICE THE RETAILER TO GIVE THEM SOME DISCOUNT TO ATTRACT THEM.

#CLUSTER 2=CUSTOMER FROM CLUSTER 2 CAN BE CONSIDERED AS LOST CUSTOMERS BECAUSE THEY HAVE SPEND LESS AND ARE LESS FREQUENT IN COMPARISON TO ANY OTHER CLUSTER
 
# CLUSTER 3= CUTOMERS FROM CLUSTER 3 SPEND LESS IN TERMS OF MONETARY VALUE AND THEY HAVE DONE MOST OF THERE SPENDING IN RECENT PERIOD. HENCE, WE CAN 
# CONSIDER THEM AS NEW CLIENTS.
 
# CLUSTER 4= CLUSTER4 CUSTOMERS ARE FREQUENTLY HIGH VALUE SPENDERS BUT THEY HAVE NOT DONE ANY TRANSACTION RECENLY. SO REATAILER CAN CONTACT THEM AND GIVE THEM SOME DISCOUNT COUPENS

# CLUSTER 5= CLUSTER 5 CUSTOMERS ARE LITTLE LESS FREQUENT AND THEY ARE THE 3RD IN TERMS OF SPENDING AND FREQUENCY ALSO BUT THEY HAVE NOT DONE ANY SPENDING
# RECENTLY    

silhouette=[]
for k in range(2,11):
    km1=KMeans(n_clusters=k,init='k-means++')
    km1.fit(df1)
    score=silhouette_score(df1,km1.labels_)
    silhouette.append(score)

print(silhouette)    







