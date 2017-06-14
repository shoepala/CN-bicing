
# coding: utf-8

# In[242]:

#import bicing data file

import json

with open('data_mon12_5am_18hrs.json') as json_data:
    data = json.load(json_data)


# In[131]:

print len(data[0]['stations'])


# In[243]:

# add labels to data 

empty_stations = []
almost_empty_stations = []
almost_full_stations = []
full_stations = []



for j in range(0,len(data)):
    for i in range(0, len(data[0]['stations'])):
        id2 = i
        data[j]['stations'][i].update({'id2':id2})   #add id2 so nodes are labelled in same way as edges
        #calc proportion of bikes in station, and add "ratio" to dict
        if (float((data[j]['stations'][i]['slots']))+float((data[j]['stations'][i]['bikes']))) == 0:
            r = "CLS"
        else: 
            r = float((data[j]['stations'][i]['bikes']))/(float((data[j]['stations'][i]['slots']))+float((data[j]['stations'][i]['bikes'])))
        data[j]['stations'][i].update({'ratio':r})
        #create groupings of stations by proportion full
        if data[j]['stations'][i]['ratio'] < 0.15:
            empty_stations.append(int(data[j]['stations'][i]['id2']))
            data[j]['stations'][i].update({'ratio_label':'empty'})
        if data[j]['stations'][i]['ratio'] >= 0.15 and data[j]['stations'][i]['ratio'] < 0.50:  
            almost_empty_stations.append(int(data[j]['stations'][i]['id2']))
            data[j]['stations'][i].update({'ratio_label':'almost_empty'})
        if data[j]['stations'][i]['ratio'] >= 0.50 and data[j]['stations'][i]['ratio'] < 0.85:
            almost_full_stations.append(int(data[j]['stations'][i]['id2']))
            data[j]['stations'][i].update({'ratio_label':'almost_full'})
        if data[j]['stations'][i]['ratio'] >= 0.85:
            full_stations.append(int(data[j]['stations'][i]['id2']))
            data[j]['stations'][i].update({'ratio_label':'full'}) 



# In[244]:

#thinking about distance and weights...
#finding distance between two coordinate points (lat&lon)

import math
import numpy as np

d = len(data[0]['stations'])

#function to find distance between 2 coordinate points
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist = radius * c

    return dist


#create symmetric matrix of distances between 2 nodes

A = np.zeros((d,d))

for i in range(0, d):
    for j in range(i+1, d):
        #print(res['stations'][i]['id']), (res['stations'][j]['id'])
        lat1 = float((data[0]['stations'][i]['latitude']))
        #print lat1
        long1 = float((data[0]['stations'][i]['longitude']))
        #print long1
        lat2 = float((data[0]['stations'][j]['latitude']))
        #print lat2
        long2 = float((data[0]['stations'][j]['longitude']))
        #print long2
        dist = (distance((lat1, long1), (lat2, long2)) )
        #print dist
        A[i, j] = dist
        A[j, i] = dist
        
print A


# In[245]:

from copy import copy, deepcopy
import numpy as np



#gaussian function
def gaussian_weights(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

print "Gaussian weights:"
for x in range(13):
    print x, gaussian_weights(x, 0, 4)

#print A[A != 0].min()
#print A[A != 0].max()
A_copy = copy(A)

#Replace elements in distances-matrix with gaussian weights
for i in range(d):
    for j in range(d):
        A_copy[i][j] = gaussian_weights(A_copy[i][j], 0, 4)
        


# In[249]:

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
get_ipython().magic(u'matplotlib inline')


def bicing_network(dataset):
    n_stations = len(dataset[0]['stations'])

    temporal_networks = []
    temporal_degrees = []
    temporal_timestamp = []
    temporal_sp = []

    for t in range(0,len(dataset)):
        G_bicing = nx.Graph()

        for x in range (0, n_stations):
            G_bicing.add_node(x)
            
        for x in range (0, n_stations):
            for y in range (x, n_stations):
                
                if dataset[t]['stations'][x]['ratio_label'] == 'empty' and dataset[t]['stations'][y]['ratio_label'] == 'full' and (x != y): 
                    w = 1 - A_copy[x,y]
                    if w <  0.5:
                        G_bicing.add_edge(x, y, weight = w)
                    
                        
                    
                   
                if dataset[t]['stations'][y]['ratio_label'] == 'empty' and dataset[t]['stations'][x]['ratio_label'] == 'full' and (x != y):
                    w = 1 - A_copy[x,y]
                    if w <  0.5:
                        G_bicing.add_edge(x, y, weight = w)
        
        G_bicing_connected = G_bicing
          
        for e in range (0, n_stations):
            if G_bicing_connected.degree(e) == 0:
                G_bicing_connected.remove_node(e)
                
        
    
        temporal_sp.append(nx.average_shortest_path_length(G_bicing_connected))
        print nx.average_shortest_path_length(G_bicing_connected)
        temporal_networks.append(G_bicing)
        temporal_degrees.append(np.average(G_bicing.degree().values()))
        temporal_timestamp.append(dataset[t]['updateTime'])
        
        #print temporal_degrees[t]
        
        #nx.write_pajek(G_bicing, "bicing.net")
    
    return temporal_networks, temporal_degrees, temporal_timestamp, temporal_sp

bicing_results = bicing_network(data)

temporal_timestamp_regular = []
for i in range(0,len(bicing_results[2])):
    temporal_timestamp_regular.append((datetime.datetime.fromtimestamp(int(bicing_results[2][i])
    ).strftime('%H:%M')))
    temporal_timestamp_regular[i] = int(temporal_timestamp_regular[i].replace(':',''))

    


# In[250]:

#plot the data 

plt.plot(temporal_timestamp_regular,bicing_results[1])
plt.xlabel('Timestamp (HourMinute)')
plt.ylabel('Average degree')
#plt.xticks(np.arange(min(temporal_timestamp_regular), max(temporal_timestamp_regular), 200))
plt.title('Average degree over time Monday 12/07/2017')
plt.show()

plt.plot(temporal_timestamp_regular,bicing_results[3])
plt.xlabel('Timestamp (HourMinute)')
plt.ylabel('Average shortest path length')
#plt.xticks(np.arange(min(temporal_timestamp_regular), max(temporal_timestamp_regular), 200))
plt.title('Average shortest path over time Monday 12/07/2017')
plt.show()


# In[ ]:




# In[124]:

from collections import defaultdict

#function to return average ratio of a station in a given day

def bicing_similarity(dataset):
    n_stations = len(dataset[0]['stations'])
    
    daily_ratio_mean = []
    daily_empty = []
    daily_full = []
    daily_almost_full = []
    daily_almost_empty = []
    ratio_dictionary = dict()
    
    for x in range(0,n_stations):
        
        for t in range(0,len(dataset)):
            ratio = data[t]['stations'][x]['ratio']
            if data[t]['stations'][x]['id2'] in ratio_dictionary:
                ratio_dictionary[data[t]['stations'][x]['id2']].append(ratio)
            else:
                ratio_dictionary[data[t]['stations'][x]['id2']] = [ratio]
            #print ratio_dictionary
            
    for y in range(0,n_stations):
        for c in range (0, len(ratio_dictionary[y])):
            if ratio_dictionary[y][c] == 'CLS':
                ratio_dictionary[y][c] = 0.0

        daily_ratio_mean.append(np.mean(ratio_dictionary[y]))
        
    for m in range(0,len(daily_ratio_mean)):
        if daily_ratio_mean[m] < 0.15:
            daily_empty.append(m)
        elif daily_ratio_mean[m] <= 0.5:
            daily_almost_empty.append(m)
        elif daily_ratio_mean[m] <= 0.85:
            daily_almost_full.append(m)
        else:
            daily_full.append(m)
            
    return daily_empty, daily_almost_empty, daily_almost_full, daily_full

daily_sim = bicing_similarity(data)

daily_e = daily_sim[0]
daily_ae = daily_sim[1]
daily_af = daily_sim[2]
daily_f = daily_sim[3]

# print daily_e
# print daily_ae
# print daily_af
# print daily_f


# In[125]:

# average split by time of day

def similarity_day_split(dataset):
    n_stations = len(dataset[0]['stations'])
    
    morning_mean = []
    midday_mean = []
    eve_mean = []
    
    morning_empty = []
    morning_full = []
    morning_almost_full = []
    morning_almost_empty = []
    
    midday_empty = []
    midday_full = []
    midday_almost_full = []
    midday_almost_empty = []
    
    eve_empty = []
    eve_full = []
    eve_almost_full = []
    eve_almost_empty = []
    
    mor_ratio_dictionary = dict()
    mid_ratio_dictionary = dict()
    eve_ratio_dictionary = dict()
    
    timestamp_u = []
    for timestamp in range(0,len(dataset)):
        timestamp_u.append(dataset[timestamp]['updateTime'])
        
    
    timestamp = []                
    for i in range(0,len(dataset)):
        timestamp.append((datetime.datetime.fromtimestamp(int(timestamp_u[i])).strftime('%H:%M')))
        timestamp[i] = int(timestamp[i].replace(':',''))
    
    for x in range(0,n_stations):
        
        for t in range(0,len(dataset)):
            
            if timestamp[t] < 1100: 
                ratio = data[t]['stations'][x]['ratio']
                if data[t]['stations'][x]['id2'] in mor_ratio_dictionary:
                    mor_ratio_dictionary[data[t]['stations'][x]['id2']].append(ratio)
                else:
                    mor_ratio_dictionary[data[t]['stations'][x]['id2']] = [ratio]  
            elif timestamp[t] < 1700: 
                ratio = data[t]['stations'][x]['ratio']
                if data[t]['stations'][x]['id2'] in mid_ratio_dictionary:
                    mid_ratio_dictionary[data[t]['stations'][x]['id2']].append(ratio)
                else:
                    mid_ratio_dictionary[data[t]['stations'][x]['id2']] = [ratio]    
            else:
                ratio = data[t]['stations'][x]['ratio']
                if data[t]['stations'][x]['id2'] in eve_ratio_dictionary:
                    eve_ratio_dictionary[data[t]['stations'][x]['id2']].append(ratio)
                else:
                    eve_ratio_dictionary[data[t]['stations'][x]['id2']] = [ratio]                    
            
    for y in range(0,n_stations):
        for c in range (0, len(mor_ratio_dictionary[y])):
            if mor_ratio_dictionary[y][c] == 'CLS':
                mor_ratio_dictionary[y][c] = 0.0
        for c in range (0, len(mid_ratio_dictionary[y])):
            if mid_ratio_dictionary[y][c] == 'CLS':
                mid_ratio_dictionary[y][c] = 0.0
        for c in range (0, len(eve_ratio_dictionary[y])):
            if eve_ratio_dictionary[y][c] == 'CLS':
                eve_ratio_dictionary[y][c] = 0.0
        
        morning_mean.append(np.mean(mor_ratio_dictionary[y]))
        midday_mean.append(np.mean(mid_ratio_dictionary[y]))
        eve_mean.append(np.mean(eve_ratio_dictionary[y]))
                
    for m in range(0,len(morning_mean)):
            if morning_mean[m] < 0.15:
                morning_empty.append(m)
            elif morning_mean[m] <= 0.5:
                morning_almost_empty.append(m)
            elif morning_mean[m] <= 0.85:
                morning_almost_full.append(m)
            else:
                morning_full.append(m)
                
    for m in range(0,len(midday_mean)):
            if midday_mean[m] < 0.15:
                midday_empty.append(m)
            elif midday_mean[m] <= 0.5:
                midday_almost_empty.append(m)
            elif midday_mean[m] <= 0.85:
                midday_almost_full.append(m)
            else:
                midday_full.append(m)
    for m in range(0,len(eve_mean)):
            if eve_mean[m] < 0.15:
                eve_empty.append(m)
            elif eve_mean[m] <= 0.5:
                eve_almost_empty.append(m)
            elif eve_mean[m] <= 0.85:
                eve_almost_full.append(m)
            else:
                eve_full.append(m)
                
    return morning_empty, morning_almost_empty, morning_almost_full, morning_full, midday_empty, midday_almost_empty, midday_almost_full, midday_full, eve_empty, eve_almost_empty, eve_almost_full, eve_full

daily_sim_split = similarity_day_split(data)

morning_e = daily_sim_split[0]
morning_ae = daily_sim_split[1]
morning_af = daily_sim_split[2]
morning_f = daily_sim_split[3]

midday_e = daily_sim_split[4]
midday_ae = daily_sim_split[5]
midday_af = daily_sim_split[6]
midday_f = daily_sim_split[7]

eve_e = daily_sim_split[8]
eve_ae = daily_sim_split[9]
eve_af = daily_sim_split[10]
eve_f = daily_sim_split[11]

print 'morning', morning_e, morning_ae, morning_af, morning_f
print 'midday', midday_e, midday_ae, midday_af, midday_f
print 'evening', eve_e, eve_ae, eve_af, eve_f


# In[241]:

import random


def bicing_heuristic(dataset):
    n_stations = len(dataset[0]['stations'])

    temporal_networks = []
    temporal_degrees = []
    temporal_timestamp = []
    temporal_sp = []

    
    G_bicing = nx.Graph()

    for x in range (0, n_stations):
        G_bicing.add_node(x) #, pos=(dataset[0]['stations'][x]['longitude'],dataset[0]['stations'][x]['latitude']))
            
    for x in range (0, n_stations):
        for y in range (x, n_stations):
                
            if dataset[0]['stations'][x]['ratio_label'] == 'empty' and dataset[0]['stations'][y]['ratio_label'] == 'full' and (x != y): 
                
                w = 1 - A_copy[x,y]
                if w <  0.2:
                    G_bicing.add_edge(x, y, weight = w)
            if dataset[0]['stations'][0]['ratio_label'] == 'empty' and dataset[0]['stations'][x]['ratio_label'] == 'full' and (x != y):
                w = 1 - A_copy[x,y]
                if w < 0.2:
                   
                    G_bicing.add_edge(x, y, weight = w)
        
    
    # picked five nodes based on the graphs
    # for each node, look around to its neighbors and pick closest one
    #print G_bicing.degree()
    
    
    for e in range (0, n_stations):
        if G_bicing.degree(e) == 0:
            G_bicing.remove_node(e)
 
    nodes = G_bicing.nodes()
    initial_station = nodes[random.randint(0,len(G_bicing.nodes())-1)]
    print G_bicing
    #print initial_station
    
    steps = 0
    distance_traveled = 0 
    closest = initial_station
    
    while G_bicing.number_of_edges() > 0 and G_bicing.degree(initial_station) > 0: 
        
        length=nx.single_source_shortest_path_length(G_bicing,initial_station, 'w')
        print length
        del length[initial_station]
        closest =  min(length, key=length.get)
        print length
        print length[closest]
        print 'c', closest
        G_bicing.remove_edge(initial_station, closest)
        
       
        steps = steps+1
        print 'sp', nx.shortest_path(G_bicing,source=initial_station,target=closest)
        distance_traveled = distance_traveled + nx.shortest_path_length(G_bicing,source=initial_station,target=closest)
        initial_station = closest


            

#     print G_bicing.degree().values()
#     print G_bicing.degree()    
    
       
    ##nx.write_pajek(G_bicing, "bicing_h2.net")
    
    
    #print G_bicing.degree().values()
        
    print G_bicing.number_of_edges()
    #print nx.average_shortest_path_length(G_bicing)
    #print G_bicing.edges()
    #nx.draw(G_bicing, nx.get_node_attributes(G_bicing, 'pos'), node_size=10)

bicing_heuristic(data)


# In[ ]:

while G_bicing.number_of_edges() > 0 and has_path(G_bicing, initial_station, closest): 


# In[ ]:

print 

