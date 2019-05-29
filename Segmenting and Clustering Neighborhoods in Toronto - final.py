#!/usr/bin/env python
# coding: utf-8

# #### Installing Beautiful Soup

# In[31]:


from bs4 import BeautifulSoup
import requests
import lxml.html as lh

import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# In[2]:


website_url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
#Create a handle, page, to handle the contents of the website
page = requests.get(website_url)
#Store the contents of the website under doc
doc = lh.fromstring(page.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')


# In[3]:


#Check the length of the first 12 rows
[len(T) for T in tr_elements[:12]]


# In[4]:


tr_elements = doc.xpath('//tr')
#Create empty list
col=[]
i=0


# In[5]:


#For each row, store each first element (header) and an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    print('%d:"%s"'%(i,name))
    col.append((name,[]))


# In[6]:


#Since out first row is the header, data is stored on the second row onwards
for j in range(1,len(tr_elements)):
    #T is our j'th row
    T=tr_elements[j]
    
    #If row is not of size 10, the //tr data is not from our table 
    if len(T)!=3:
        break
    
    #i is the index of our column
    i=0
    
    #Iterate through each element of the row
    for t in T.iterchildren():
        data=t.text_content() 
        #Check if row is empty
        if i>0:
        #Convert any numerical value to integers
            try:
                data=int(data)
            except:
                pass
        #Append the data to the empty list of the i'th column
        col[i][1].append(data)
        #Increment i for the next column
        i+=1


# In[7]:


[len(C) for (title,C) in col]


# In[8]:


Dict={title:column for (title,column) in col}
df= pd.DataFrame(Dict)


# In[9]:


df.head()


# In[10]:


new_order = [-1,0,1]
df = df[df.columns[new_order]]
df.head()


# In[11]:


df = df[df.Borough != 'Not assigned']


# In[12]:


df.shape


# In[13]:


df = df.reset_index(drop=True)


# In[14]:


df.head()


# In[15]:


df.columns.values


# In[16]:


df.rename(columns={'Neighbourhood\n': 'Neighbourhood'}, inplace=True)


# In[17]:


df['Neighbourhood'] = df['Neighbourhood'].map(lambda x: x.rstrip('\n'))


# In[18]:


df.head()


# In[19]:


# If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough. 
# So for the 9th cell in the table on the Wikipedia page, the value of the Borough and the Neighborhood columns will be Queen's Park.

for index, row in df.iterrows():
     if row['Neighbourhood'] == "Not assigned":
        row['Neighbourhood'] = row['Borough']


# In[20]:


# More than one neighborhood can exist in one postal code area. 
# For example, in the table on the Wikipedia page, you will notice that M5A is listed twice and has two neighborhoods: Harbourfront and Regent Park. 
# These two rows will be combined into one row with the neighborhoods separated with a comma as shown in row 11 in the above table.

df = df.groupby(['Postcode','Borough'])['Neighbourhood'].apply(', '.join).reset_index()


# In[21]:


df.shape


# In[23]:


df_geocode=pd.read_csv('Geospatial_Coordinates.csv')
df_geocode.head()


# In[24]:


df = pd.merge(df, df_geocode, how='inner', left_on = 'Postcode', right_on = 'Postal Code')
df.head()


# In[26]:


df = df[['Postcode', 'Borough', 'Neighbourhood','Latitude', 'Longitude']]


# In[28]:


df.head()


# In[29]:


df.shape


# In[32]:


address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates of Toronto are {}, {}.'.format(latitude, longitude))


# In[35]:


# create map of Manhattan using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, label in zip(df['Latitude'], df['Longitude'], df['Postcode']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[42]:


# set number of clusters
kclusters = 4

toronto_clustering = df[['Latitude', 'Longitude']]
# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[51]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df['Latitude'], df['Longitude'], df['Neighbourhood'], df['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




