#!/usr/bin/env python
# coding: utf-8

# <div style="text-align:center"> <h1><font color='SlateBlue'>CHENNAI HOUSE SALES PRICE PREDICTION</font><h2></div>

# # Problem statement:
#                  Real estate transactions are quite opaque sometimes and it may be difficult for a newbie to know the fair price of any given home. Thus, multiple real estate websites have the functionality to predict the prices of houses given different features regarding it. Such forecasting models will help buyers to identify a fair price for the home and also give insights to sellers as to how to build homes that fetch them more money.  Chennai house sale price data is shared here and the participants are expected to build a sale price prediction model that will aid the customers to find a fair price for their homes and also help the sellers understand what factors are fetching more money for the houses?
#                   
# ### Explanation of column values:  
# **1.SALE_COND** <br>
# * ***Family***: Sales happens within your family members. 
# * ***Partial***: Selling only part of the builing.
# * ***AdjLand***: Selling the adjacent building. 
# * ***Normal Sale***: Just a normal sale.
# * ***AbNormal***: If above conditons not suited
# 
# **2.PARK_FACIL** <br>
# * ***Yes***: If parking facility available.
# * ***No***: If parking facility not available.
# 
# **3.BUILDTYPE** <br>
# * ***House***: If the builing is house.
# * ***Commercial***:If the building is for commercial purpose.
# * ***Others***:If anything not comes under above two.
# 
# **4.UTILITY_AVAIL** <br>
# * ***AllPub***: If all facilities available
# * ***ELO***: If ELO facility available
# * ***NoSewr***: If there is no sewage system.
# 
# **5.STREET** <br>
# * ***Paved***: If it has a proper road.
# * ***Gravel***: If it has a gravel road.
# * ***No Access***: If it has no access to the road.
# 
# **6.MZZONE** <br>
# * ***A***: Agricultural Land
# * ***RH***: Residential High
# * ***RM***: Residential Medium
# * ***RL***: Residential Low
# * ***I***: Industial Land
# * ***C***: Commercial Land

# <div style="text-align:left"> <h1><font color='red'>Start of codes:</font><h2></div>

# In[159]:


#importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[160]:


#loading the csv file in pandas datadrame.
df=pd.read_csv('train-chennai-sale.csv')
pd.set_option('display.max_columns',None)
df


# In[161]:


df.shape


# ## Data cleaning:
# * Check for duplicates
# * change into correct datatypes
# * Imputation of missing values
# * Correction of spelling mistakes

# In[162]:


#dropping the duplicates
df=df.drop_duplicates()


# In[163]:


#checking for null values
df.isnull().sum()


# In[164]:


#Verifying the datatypes
df.info()


# In[165]:


df.describe()


# In[166]:


plt.figure(figsize=(30, 20), dpi=150)

plt.subplot(6,2,1)
df.boxplot(column = 'INT_SQFT')

plt.subplot(6,2,2)
df.boxplot(column = 'DIST_MAINROAD')

plt.subplot(6,2,3)
df.boxplot(column = 'N_BEDROOM')

plt.subplot(6,2,4)
df.boxplot(column = 'N_BATHROOM')

plt.subplot(6,2,5)
df.boxplot(column = 'N_ROOM')

plt.subplot(6,2,6)
df.boxplot(column = 'QS_ROOMS')

plt.subplot(6,2,7)
df.boxplot(column = 'QS_BATHROOM')

plt.subplot(6,2,8)
df.boxplot(column = 'QS_BEDROOM')

plt.subplot(6,2,9)
df.boxplot(column = 'QS_OVERALL')

plt.subplot(6,2,10)
df.boxplot(column = 'REG_FEE')

plt.subplot(6,2,11)
df.boxplot(column = 'COMMIS')

plt.subplot(6,2,12)
df.boxplot(column = 'SALES_PRICE')

plt.show()


# In[167]:


for col in df.columns:
    print(col,'Unique count is :',len(df[col].unique()))
    print()


# In[168]:


#Checking the unique values for categorical feautures
for val in df.columns:
    if df[val].dtype=='object':
        print(val,'unique values are',df[val].unique())
        print()


# ## General observation from the dataframe:<br>
# * There is no duplicates in the dataset.
# * 53 null values identified.
# * Datatype of **DATE_SALE, DATE_BUILD, N_BEDROOM and N_BATHROOM** are wrong.
# * Spelling mistakes in numerous categorical column values.
# * Age of building can be calculated with **DATE_SALE and DATE_BUILD**.
# * We have four feautures of masked values.

# In[169]:


#Filling the null values.
#N_BEDROOM has 1 null value we can use mode for this as its a discrete numerical variable.
#N_BATHROOM have 5 null values we can use mode for this as its a discrete numerical variable.
df['N_BEDROOM'].fillna(df['N_BEDROOM'].mode()[0], inplace=True)
df['N_BATHROOM'].fillna(df['N_BATHROOM'].mode()[0],inplace=True)


#For QS_OVERALL , the values can be filled by the mean of QS_ROOMS, QS_BATHROOM and QS_BEDROOM
df['QS_OVERALL'].fillna(((df['QS_ROOMS']+df['QS_BATHROOM']+df['QS_BEDROOM'])/3),inplace=True)


# In[170]:


#Correcting the datatypes of the feautures.
df['DATE_SALE']= pd.to_datetime(df['DATE_SALE'])
df['DATE_BUILD']= pd.to_datetime(df['DATE_BUILD'])
df=df.astype({"N_BEDROOM":'int',"N_BATHROOM":'int'})


# In[171]:


df.info()


# In[172]:


#Feauture engineering
#Calculating the age of the building
df['BUILD_AGE'] = pd.DatetimeIndex(df['DATE_SALE']).year - pd.DatetimeIndex(df['DATE_BUILD']).year


# In[173]:


for col in df.columns:
    print(col,'Unique count is :',len(df[col].unique()))
    print()


# In[174]:


for val in df.columns:
    if df[val].dtype=='object':
        print(val,'unique values are',df[val].unique())
        print()


# ### Dropping of feautures:
# * **PRT_ID** has only unique values so we can drop it.
# * **REG_FEE** and **COMMIS** values, we won't be knowing this before hand when we buy a house,so we can drop it.
# * After calculating the age of the building we can drop the **DATE_SALE** and **DATE_BUILD**.

# In[175]:


#dropping the unnecessary feautures.
del df['PRT_ID']
del df['REG_FEE']
del df['COMMIS']
del df['DATE_BUILD']
del df['DATE_SALE']


# In[176]:


#Correcting the spelling mistakes in the column values.
df['AREA']= df['AREA'].replace(['Karapakam'],'Karapakkam')
df['AREA']= df['AREA'].replace(['Ana Nagar','Ann Nagar'],'Anna Nagar')
df['AREA']= df['AREA'].replace(['Adyr'],'Adyar')
df['AREA']= df['AREA'].replace(['Velchery'],'Velachery')
df['AREA']= df['AREA'].replace(['Chrompt','Chrmpet','Chormpet'],'Chrompet')
df['AREA']= df['AREA'].replace(['KKNagar'],'KK Nagar')
df['AREA']= df['AREA'].replace(['T Nagar'],'TNagar')


# In[177]:


df['SALE_COND']= df['SALE_COND'].replace(['Ab Normal'],'AbNormal')
df['SALE_COND']= df['SALE_COND'].replace(['Partiall','PartiaLl'],'Partial')
df['SALE_COND']= df['SALE_COND'].replace(['Adj Land'],'AdjLand')


# In[178]:


df['PARK_FACIL']= df['PARK_FACIL'].replace(['Noo'],'No')


# In[179]:


df['BUILDTYPE']= df['BUILDTYPE'].replace(['Comercial'],'Commercial')
df['BUILDTYPE']= df['BUILDTYPE'].replace(['Other'],'Others')


# In[180]:


df['UTILITY_AVAIL']= df['UTILITY_AVAIL'].replace(['All Pub'],'AllPub')
df['UTILITY_AVAIL']= df['UTILITY_AVAIL'].replace(['NoSeWa'],'NoSewr ')


# In[181]:


df['STREET']= df['STREET'].replace(['Pavd'],'Paved')
df['STREET']= df['STREET'].replace(['NoAccess'],'No Access')


# In[182]:


for val in df.columns:
    if df[val].dtype=='object':
        print(val,'unique values are',df[val].unique())
        print()


# In[183]:


#Reindexing the column for easy data visualization.
df=df.reindex(columns=['AREA', 'SALE_COND', 'PARK_FACIL',
       'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE', 'BUILD_AGE', 
       'INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM','N_BATHROOM', 'N_ROOM', 
       'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 
       'SALES_PRICE',])


# # EDA:
# * For each categorical feauture we will plot both the distribution and categorical vs target plot.
# * For each numerical feauture we will plot both the distribution and categorical vs target plot. In addition to that we will plot the box plot to detect the outliers.

# In[184]:


df


# In[185]:


#Heatmap to get the linear relationship between different feautures.
plt.figure(figsize=(12,7), dpi=150)
sns.heatmap(df.corr(method='pearson'), cbar=False, annot=True, fmt='.1f', linewidth=0.2, cmap='coolwarm');


# ### Interpretation of heatmap:
# * There is a better linear relationship between **INT_SQFT** and **SALES_PRICE**.
# * There is a better linear relationship between **N_ROOM** and **SALES_PRICE**.
# * There is a slight linear relationship between **N_BEDROOM** and **SALES_PRICE**.
# 
# 
# 

# In[186]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['AREA'], linewidth=0,kde=True)
plt.xticks(rotation=45)
plt.title('Distribution In Terms Of AREA')

plt.subplot(2,2,2)
sns.barplot(x=df['AREA'],y=df['SALES_PRICE'],order=df.groupby('AREA')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['AREA'])
plt.xticks(rotation=45)
plt.title('AREA vs SALES_PRICE')

plt.show()


# ### Interpretation:
# * **Chrompet** and **Karapakkam** have more number of distribution of houses compared to other areas.
# * There is a good linear relationship between **AREA** and **SALES_PRICE**, so we can follow the label encoding here.
# * **SALES_PRICE** is least in **Karapakkam** and high in **TNagar**.

# In[187]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['SALE_COND'], linewidth=0,kde=True)
plt.xticks(rotation=45)
plt.title('Distribution In Terms Of SALE_COND')

plt.subplot(2,2,2)
sns.barplot(x=df['SALE_COND'],y=df['SALES_PRICE'],order=df.groupby('SALE_COND')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['SALE_COND'])
plt.xticks(rotation=45)
plt.title('SALE_COND vs SALES_PRICE')

plt.show()


# ### Interpretation:
# * More or less equally distributed.
# * There is a very slight linear relationship between **SALE_COND** and **SALES_Price**.
# * As relationship is very negligible we can drop this feauture .

# In[188]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['PARK_FACIL'], linewidth=0,kde=True)
plt.xticks(rotation=45)
plt.title('Distribution In Terms Of PARK_FACIL')

plt.subplot(2,2,2)
sns.barplot(x=df['PARK_FACIL'],y=df['SALES_PRICE'])
plt.xticks(rotation=45)
plt.title('PARK_FACIL vs SALES_PRICE')

plt.show()


# ### Interpretation:
# * More or less equally distributed.
# * There is a good linear relationship between **PARK_FACIL** and **SALES_PRICE**.
# * we can use the label encoding here.

# In[189]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['BUILDTYPE'], linewidth=0,kde=True)
plt.xticks(rotation=45)
plt.title('Distribution In Terms Of BUILDTYPE')

plt.subplot(2,2,2)
sns.barplot(x=df['BUILDTYPE'],y=df['SALES_PRICE'],order=df.groupby('BUILDTYPE')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['BUILDTYPE'])
plt.xticks(rotation=45)
plt.title('BUILDTYPE vs SALES_PRICE')

plt.show()


# ### Interpretation:
# * More or less equally distributed.
# * There is no linear relationship between **BUILDTYPE** and **SALES_PRICE**.
# * we can use the onehot encoding here.

# In[190]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['UTILITY_AVAIL'], linewidth=0,kde=True)
plt.xticks(rotation=45)
plt.title('Distribution In Terms Of UTILITY_AVAIL')

plt.subplot(2,2,2)
sns.barplot(x=df['UTILITY_AVAIL'],y=df['SALES_PRICE'],order=df.groupby('UTILITY_AVAIL')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['UTILITY_AVAIL'])
plt.xticks(rotation=45)
plt.title('UTILITY_AVAIL vs SALES_PRICE')

plt.show()


# ### Interpretation:
# * NoSewr property has more distribution.
# * There is a good linear relationship between **UTILITY_AVAIL** and **SALES_PRICE**.
# * we can use the label encoding here.

# In[191]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['STREET'], linewidth=0,kde=True)
plt.xticks(rotation=45)
plt.title('Distribution In Terms Of STREET')

plt.subplot(2,2,2)
sns.barplot(x=df['STREET'],y=df['SALES_PRICE'],order=df.groupby('STREET')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['STREET'])
plt.xticks(rotation=45)
plt.title('STREET vs SALES_PRICE')

plt.show()


# ### Interpretation:
# * More or less equally distributed.
# * There is a good linear relationship between **STREET** and **SALES_PRICE**.
# * we can use the label encoding here.

# In[192]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['MZZONE'], linewidth=0,kde=True)
plt.xticks(rotation=45)
plt.title('Distribution In Terms Of MZZONE')

plt.subplot(2,2,2)
sns.barplot(x=df['MZZONE'],y=df['SALES_PRICE'],order=df.groupby('MZZONE')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['MZZONE'])
plt.xticks(rotation=45)
plt.title('MZZONE vs SALES_PRICE')

plt.show()


# ### Interpretation:
# * More or less equally distributed.
# * There is a good linear relationship between **MZZONE** and **SALES_PRICE** (but from I to RH there is a jump).
# * we can use the label encoding here (so we have to skip one value from I to H).

# In[193]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.distplot(df['BUILD_AGE'])
plt.title('Distribution In Terms Of BUILD_AGE')

plt.subplot(2,2,2)
sns.scatterplot(data=df,x=df['BUILD_AGE'],y=df['SALES_PRICE'])
plt.title('BUILD_AGE vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['BUILD_AGE'])
plt.title('Box plot of BUILD_AGE')

plt.show()

cor=df['BUILD_AGE'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * Most densely distributed property are between 0-40 years of age.
# * There is a linear relationship between **BUILD_AGE** and **SALES_PRICE**.
# * There is no outlier in the **BUILD_AGE** feauture.

# In[194]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.distplot(df['INT_SQFT'])
plt.title('Distribution In Terms Of INT_SQFT')

plt.subplot(2,2,2)
sns.scatterplot(data=df,x=df['INT_SQFT'],y=df['SALES_PRICE'])
plt.title('INT_SQFT vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['BUILD_AGE'])
plt.title('Box plot of INT_SQFT')

plt.show()

cor=df['INT_SQFT'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * There is a good linear relationship between **INT_SQFT** and **SALES_PRICE**.
# * There is no outlier in the **INT_SQFT** feauture.

# In[195]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.distplot(df['DIST_MAINROAD'])
plt.title('Distribution In Terms Of DIST_MAINROAD')

plt.subplot(2,2,2)
sns.scatterplot(data=df,x=df['DIST_MAINROAD'],y=df['SALES_PRICE'])
plt.title('DIST_MAINROAD vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['BUILD_AGE'])
plt.title('Box plot of DIST_MAINROAD')

plt.show()

cor=df['DIST_MAINROAD'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * More or less property are equally distributed .
# * There is no linear relationship between **DIST_MAINROAD** and **SALES_PRICE**.
# * There is no outlier in the **DIST_MAINROAD** feauture.

# In[196]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['N_BEDROOM'], linewidth=0,kde=True)
plt.title('Distribution In Terms Of N_BEDROOM')

plt.subplot(2,2,2)
sns.barplot(x=df['N_BEDROOM'],y=df['SALES_PRICE'],order=df.groupby('N_BEDROOM')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['N_BEDROOM'])
plt.title('N_BEDROOM vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['N_BEDROOM'])
plt.title('Box plot of N_BEDROOM')

plt.show()

cor=df['N_BEDROOM'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * Most of the property has 1 bedroom and 2 berooms.
# * There is a linear relationship between **N_BEDROOM** and **SALES_PRICE**.
# * 4 is a outlier here, but we should not remove it for this business case.

# In[197]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['N_BATHROOM'], linewidth=0,kde=True)
plt.title('Distribution In Terms Of N_BATHROOM')

plt.subplot(2,2,2)
sns.barplot(x=df['N_BATHROOM'],y=df['SALES_PRICE'],order=df.groupby('N_BATHROOM')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['N_BATHROOM'])
plt.title('N_BATHROOM vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['N_BATHROOM'])
plt.title('Box plot of N_BATHROOM')

plt.show()

cor=df['N_BATHROOM'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * Most of the property has 1 bathroom.
# * There is a linear relationship between **N_BATHROOM** and **SALES_PRICE**.
# * 2 is a outlier here, but we should not remove it for this business case.

# In[198]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.histplot(df['N_ROOM'], linewidth=0,kde=True)
plt.title('Distribution In Terms Of N_ROOM')

plt.subplot(2,2,2)
sns.barplot(x=df['N_ROOM'],y=df['SALES_PRICE'],order=df.groupby('N_ROOM')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['N_ROOM'])
plt.title('N_ROOM vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['N_ROOM'])
plt.title('Box plot of N_ROOM')

plt.show()

cor=df['N_ROOM'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * Most of the property has 3,4,5 rooms.
# * There is a linear relationship between **N_ROOM** and **SALES_PRICE**.
# * 6 is a outlier here, but we should not remove it for this business case.

# In[199]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.distplot(df['QS_ROOMS'])
plt.title('Distribution In Terms Of QS_ROOMS')

plt.subplot(2,2,2)
sns.scatterplot(data=df,x=df['QS_ROOMS'],y=df['SALES_PRICE'])
plt.title('QS_ROOMS vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['QS_ROOMS'])
plt.title('Box plot of QS_ROOMS')

plt.show()

cor=df['QS_ROOMS'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * There is no linear relationship between **QS_ROOMS** and **SALES_PRICE**.
# * There is no outlier here.

# In[200]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.distplot(df['QS_BATHROOM'])
plt.title('Distribution In Terms Of QS_BATHROOM')

plt.subplot(2,2,2)
sns.scatterplot(data=df,x=df['QS_BATHROOM'],y=df['SALES_PRICE'])
plt.title('QS_BATHROOM vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['QS_BATHROOM'])
plt.title('Box plot of QS_BATHROOM')

plt.show()

cor=df['QS_BATHROOM'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * There is no linear relationship between **QS_BATHROOM** and **SALES_PRICE**.
# * There is no outlier here.

# In[201]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.distplot(df['QS_BEDROOM'])
plt.title('Distribution In Terms Of QS_BEDROOM')

plt.subplot(2,2,2)
sns.scatterplot(data=df,x=df['QS_BEDROOM'],y=df['SALES_PRICE'])
plt.title('QS_BATHROOM vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['QS_BEDROOM'])
plt.title('Box plot of QS_BEDROOM')

plt.show()

cor=df['QS_BEDROOM'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * There is no linear relationship between **QS_BEDROOM** and **SALES_PRICE**.
# * There is no outlier here.

# In[202]:


plt.figure(figsize=(20, 10), dpi=150)

plt.subplot(2,2,1)
sns.distplot(df['QS_OVERALL'])
plt.title('Distribution In Terms Of QS_OVERALL')

plt.subplot(2,2,2)
sns.scatterplot(data=df,x=df['QS_OVERALL'],y=df['SALES_PRICE'])
plt.title('QS_OVERALL vs SALES_PRICE')

plt.subplot(2,2,3)
plt.boxplot(df['QS_OVERALL'])
plt.title('Box plot of QS_OVERALL')

plt.show()

cor=df['QS_OVERALL'].corr(df['SALES_PRICE'])
print('The corr between feauture and target :',cor)


# ### Interpretation:
# * There is no linear relationship between **QS_BEDROOM** and **SALES_PRICE**.
# * There is no outlier here.

# ## OVERALL OBSERVATION
# * There is no notable outlier which can affect the efficiency of the model.
# * **QS_OVERALL,DIST_MAINROAD,QS_ROOMS',QS_BEDROOM,QS_BATHROOM** should be dropped as it don't have any relationship with target feauture.
# * With increase in no of bedrooms , salesprice also increasing.
# * Commercial building has highest salesprice followed by House and Others.
# * There is a linear relationship between Area, MZ Zone, N ROOM, N BEDROOM, N BATHROOM, STREET,SALE COND and TOTAL SALE PRICE
# * For AREA, STREET, MZ ZONE, and SALE COND, we apply ordinal label encoding.
# * Although there is no linear relationship, BUILDTYPE does impact SALE PRICE. Therefore, for this column, we choose one hot encoding.
# 
# 
# 

# In[203]:


df.drop(['QS_OVERALL','DIST_MAINROAD','QS_ROOMS','QS_BEDROOM','QS_BATHROOM'],axis=1,inplace=True)


# In[204]:


for val in df.columns:
    if df[val].dtype=='object':
        print(val,'unique values are',df[val].unique())
        print()


# ### Encoding the categorical variables:
# * We can map the categorical feauture values which follows the linear relationship with label encoding.
# * For feauture which not following linear relationship, we will go with one hot encoding.

# In[205]:


#Label encoding.
df['AREA']= df['AREA'].map({'Karapakkam': 0,
                           'Adyar': 1, 
                           'Chrompet' : 2,
                           'Velachery' : 3,
                           'KK Nagar' : 4, 
                           'Anna Nagar' : 5,
                           'TNagar' : 6})

df['PARK_FACIL'] = df['PARK_FACIL'].map({'Yes':1,
                                       'No':0})

df['UTILITY_AVAIL'] = df['UTILITY_AVAIL'].map({'ELO' : 0,  
                                             'NoSewr ' : 1,
                                             'AllPub' : 2})
                                          
df['STREET'] = df['STREET'].map({'No Access' : 0,
                               'Paved' : 1, 
                               'Gravel' : 2})


df['MZZONE'] = df['MZZONE'].map({'A' : 0,
                               'C' : 1,
                               'I' : 2,
                               'RH' : 4,#there is a break in a linear relaationship, so avoid it we skipping one number)
                               'RL' : 5,
                               'RM' : 6})

df['SALE_COND'] = df['SALE_COND'].map({'Partial' : 0,
                                'Family':1,
                                'AbNormal':2,
                                'Normal Sale':3,
                                'AdjLand':4})


# In[206]:


#One hot encoding
df = pd.get_dummies(df, columns=['BUILDTYPE'])


# In[207]:


df


# ### Preparing the data:

# In[208]:


#Reindexing the column.
df=df.reindex(columns=['AREA','SALE_COND','PARK_FACIL',
       'UTILITY_AVAIL', 'STREET', 'MZZONE', 'BUILD_AGE', 
       'INT_SQFT', 'N_BEDROOM','N_BATHROOM', 'N_ROOM', 
       'BUILDTYPE_Commercial', 'BUILDTYPE_House', 'BUILDTYPE_Others', 
       'SALES_PRICE',])


# In[209]:


#Initialising the values
X = df.drop(['SALES_PRICE'],axis=1)
y = df['SALES_PRICE']

#Splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)


# In[210]:


X_train


# ### Scaling the data

# In[211]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)


# In[212]:


x_train


# # Modelling

# ## LINEAR REGRESSION

# In[213]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
#print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_pred))


# In[214]:


y_pred = regressor.predict(x_test)
y_pred


# In[215]:


res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
res


# In[216]:


lr_met = metrics.r2_score(y_test,y_pred)
print('R2- SCORE FOR LINEAR REGRESSION:', round((lr_met*100),2),"%")


# ## K-NEAREST NEIGHBOUR MODEL

# In[217]:


from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')

MSE = []
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
k=model.best_params_


# In[218]:


for i,j in k.items():
    knn = KNeighborsRegressor(j) 
    knn.fit(X_train,y_train)
    print("K value  : " , j, " score : ", metrics.r2_score(y_test,y_pred))   


# In[219]:


KNN=KNeighborsRegressor()
KNN.fit(x_train,y_train)


# In[220]:


y_pred_knn = KNN.predict(x_test)
y_pred_knn


# In[221]:


result_knn= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_knn})
result_knn


# In[222]:


KNN_met = metrics.r2_score(y_test,y_pred_knn)
print('R2- SCORE FOR KNN MODEL:', round((KNN_met*100),2),"%")


# ## DECISION TREE MODEL

# In[223]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import numpy as np

for depth in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,100]:
    dt = DecisionTreeRegressor(max_depth=depth)
    dt.fit(x_train, y_train) # the model is trained
    valAccuracy = cross_val_score(dt, x_train, y_train, cv=10, scoring = make_scorer(metrics.r2_score))
    print("DEPTH: ",depth,"R2-Score: ",np.mean(valAccuracy))


# In[224]:


dt = DecisionTreeRegressor(max_depth=30)
dt.fit(x_train, y_train)


# In[225]:


y_pred_dt = dt.predict(x_test)
y_pred_dt


# In[226]:


res_dt=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred_dt})
res_dt


# In[227]:


dt_met = metrics.r2_score(y_test,y_pred_dt)
print('R2- SCORE FOR DECISION TREE MODEL:', round((dt_met*100),2),"%")


# ## RANDOM FOREST MODEL

# In[228]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators= 150, max_depth = 10, max_features='sqrt')
rf.fit(x_train, y_train)


# In[229]:


y_pred_rf = rf.predict(x_test)


# In[230]:


res_rf= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
res_rf


# In[231]:


rf_met = metrics.r2_score(y_test,y_pred_rf)
print('R2- SCORE FOR RANDOM FOREST MODEL :', round((rf_met*100),2),"%")


# ## XG-BOOST MODEL

# In[232]:


import xgboost as xgb
for lr in [0.01,0.02,0.03,0.04,0.05,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.5,0.7,1]:
    model = xgb.XGBRegressor(learning_rate = lr, n_estimators=100, verbosity = 0)
    model.fit(x_train, y_train)
    print("Learning rate : ", lr, " Train score : ", model.score(x_train,y_train), " Test score : ", model.score(x_test,y_test))


# In[233]:


xg = xgb.XGBRegressor(learning_rate = 0.15, n_estimators=100, verbosity = 0)
xg.fit(x_train, y_train)
y_pred_xg = xg.predict(x_test)


# In[234]:


xg_met = metrics.r2_score(y_test,y_pred_xg)
print('R2- SCORE FOR RANDOM FOREST MODEL :', round((xg_met*100),2),"%")


# ## Finding the best model

# In[235]:


r2={'R2_score':['LR', 'KNN', 'DT', 'RF',"XG"],
        'score':[92.46, 94.54, 97.42, 97.87,99.68]}
R2_df=pd.DataFrame(r2)
R2_df


# ## Interpretation:
# * From this we can interpret that **XG-BOOST** model gives highest score than others.
# * Score of **XG-BOOST** model is 99.68%.

# #  CONCLUSION:

# * We can conclude from this project that many feautures plays major roles in setting the salesprice of the chennai houses.
# * We started with understanding the data and cleaned the data to remove duplicates and imputing the missing values.
# * we spent more amount of time in exploratory data analysis to find this insights , which helped to choose the feautures which were required to train the model.
# * Most imprortantly area , sqft, no of bedromm , type of building played a major role in setting the price.
# * Then we trained the data with 5 different models.
# * we found thE **XG-BOOST** model gave the highest score of 0.9968.

# In[ ]:




