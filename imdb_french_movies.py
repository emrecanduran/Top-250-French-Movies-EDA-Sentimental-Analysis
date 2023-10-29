#!/usr/bin/env python
# coding: utf-8

# ### Important Note:
# The "Top 250 French Movies" dataset comprises information on the highest-rated French movies according to user ratings on various platforms. This dataset contains 250 unique French movies that have garnered critical acclaim and popularity among viewers. Each movie is associated with essential details, including its rank, title, release year, duration, genre, IMDb rating, image source link, and a brief description.
# 
# This dataset is intended for learning, research, and analysis purposes. The movie ratings and details provided in the dataset are based on publicly available information at the time of scraping. As IMDb ratings and movie information may change over time, it is essential to verify and update the data for the latest information.
# 
# By using this dataset, you acknowledge that the accuracy and completeness of the information cannot be guaranteed, and you assume responsibility for any analysis or decision-making based on the data. Additionally, please adhere to IMDb's terms of use and copyright policies when using the data for any public dissemination or commercial purposes.
# 
# Data Analysis Tasks:
# 
# 1.Exploratory Data Analysis (EDA):
# Explore the distribution of movies by genres, release years, and IMDb ratings. Visualize the top-rated French movies and their IMDb ratings using bar charts or histograms.
# 
# 2.Year-wise Trends:
# Observe trends in French movie production over the years using line charts or area plots. Analyze if there's any correlation between release year and IMDb ratings.
# 
# 3.Word Cloud Analysis:
# Create word clouds from movie descriptions to visualize the most common words and themes among the top-rated French movies. This can provide insights into popular topics and genres.
# 
# 4.Network Analysis:
# Build a network graph connecting French movies that share common actors or directors. Analyze the interconnectedness of movies based on their production teams.
# 
# Machine Learning Tasks:
# 
# 1.Movie Recommendation System:
# Implement a content-based recommendation system that suggests French movies based on similarities in genre, release year, and IMDb ratings. Use techniques like cosine similarity or Jaccard similarity to measure movie similarities.
# 
# 2.Movie Genre Classification:
# Build a multi-class classification model to predict the genre of an French movie based on its description. Utilize Natural Language Processing (NLP) techniques like text preprocessing, TF-IDF, or word embeddings. Use classifiers like Logistic Regression, Naive Bayes, or Support Vector Machines.
# 
# 3.Movie Sentiment Analysis:
# Perform sentiment analysis on movie descriptions to determine the overall sentiment (positive, negative, neutral) of each movie. Use sentiment lexicons or pre-trained sentiment analysis models.
# 
# 4.Movie Rating Prediction:
# Develop a regression model to predict the IMDb rating of an French movie based on features like genre, release year, and description sentiment. Employ regression algorithms like Linear Regression, Decision Trees, or Random Forests.
# 
# 5.Movie Clustering:
# Apply unsupervised clustering algorithms to group French movies with similar attributes. Use features like genre, IMDb rating, and release year to identify movie clusters. Experiment with algorithms like K-means clustering or hierarchical clustering.
# 
# Important Note:
# Ensure that the data is appropriately preprocessed and encoded for machine learning tasks. Handle any missing values, perform feature engineering, and split the dataset into training and testing sets. Evaluate the performance of each machine learning model using appropriate metrics such as accuracy, precision, recall, or Mean Squared Error (MSE) depending on the task.
# 
# It is crucial to remember that the performance of machine learning models may vary based on the dataset's size and quality. Interpret the results carefully and consider using cross-validation techniques to assess model generalization.
# 
# Lastly, please adhere to IMDb's terms of use and any applicable data usage policies while conducting data analysis and implementing machine learning models with this dataset.

# In[1]:


import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import networkx as nx
from collections import Counter


# In[2]:


# Load the data
imdb = pd.read_csv('imdb.csv')


# In[3]:


# Dataset has 250 entries and 15 columns. Some columns have missing values
imdb.info()


# In[4]:


# Drop the useless columns 
imdb = imdb.drop(['web-scraper-order', 'web-scraper-start-url', 'Image-src', 'Votes', 'Gross'], axis=1)


# In[5]:


imdb.head()


# In[6]:


# Check duplicates
imdb.duplicated().sum()


# ### Missing values

# In[7]:


count = imdb.isnull().sum().sort_values(ascending=False)
percentage = ((imdb.isnull().sum()/len(imdb)*100)).sort_values(ascending=False)
missing_data = pd.concat([count, percentage], axis=1, 
keys = ['Count', 'Percentage'])

print('Count and percantage of missing values for the columns:')

missing_data


# In[8]:


print(imdb['Type'].value_counts())
print(imdb['MetaScore'].value_counts())


# ### Weighted Imputation

# In[9]:


# Calculate the weights based on value frequencies for 'Type'
value_counts_type = imdb['Type'].value_counts()
weights_type = value_counts_type / value_counts_type.sum()

# Calculate the weights based on value frequencies for 'MetaScore'
value_counts_metascore = imdb['MetaScore'].value_counts()
weights_metascore = value_counts_metascore / value_counts_metascore.sum()

# Create an array of unique values with their corresponding weights for 'Type'
unique_values_type = weights_type.index.tolist()
unique_weights_type = weights_type.values.tolist()

# Create an array of unique values with their corresponding weights for 'MetaScore'
unique_values_metascore = weights_metascore.index.tolist()
unique_weights_metascore = weights_metascore.values.tolist()

# Create masks for missing values in both columns
type_mask = imdb['Type'].isnull()
metascore_mask = imdb['MetaScore'].isnull()

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random imputations based on weights for 'Type'
imputations_type = np.random.choice(unique_values_type, size=type_mask.sum(), p=unique_weights_type)

# Generate random imputations based on weights for 'MetaScore'
imputations_metascore = np.random.choice(unique_values_metascore, size=metascore_mask.sum(), p=unique_weights_metascore)

# Create a copy of the DataFrame to avoid modifying the original data
imdb_imputed = imdb.copy()

# Assign the imputations to the missing values in the copy DataFrame for 'Type'
imdb_imputed.loc[type_mask, 'Type'] = imputations_type

# Assign the imputations to the missing values in the copy DataFrame for 'MetaScore'
imdb_imputed.loc[metascore_mask, 'MetaScore'] = imputations_metascore

# Verify the imputed values for 'Type'
imputed_values_type = imdb_imputed['Type'].loc[type_mask]
print("Imputed values for 'Type':")
print(imputed_values_type)

# Verify the imputed values for 'MetaScore'
imputed_values_metascore = imdb_imputed['MetaScore'].loc[metascore_mask]
print("Imputed values for 'MetaScore':")
print(imputed_values_metascore)

# Update the original DataFrame with the imputed values for 'Type'
imdb['Type'].update(imdb_imputed['Type'])

# Update the original DataFrame with the imputed values for 'MetaScore'
imdb['MetaScore'].update(imdb_imputed['MetaScore'])


# In[10]:


print(imdb['Type'].value_counts())
print(imdb['MetaScore'].value_counts())


# In[11]:


imdb.isnull().sum()


# ### Summary Statistics

# In[12]:


# Describe 
imdb.describe(include='all').T


# In[13]:


imdb['Year'].unique()


# In[14]:


# Extract and clean the years using regular expressions
imdb['Year'] = imdb['Year'].apply(lambda x: re.search(r'\d{4}', x).group() if re.search(r'\d{4}', x) else None)

print(imdb['Year'].unique())


# In[15]:


# Split the 'Genre' column into multiple genres
imdb['Genres'] = imdb['Genre'].str.split(', ')

# Create a DataFrame to count the frequency of each genre
genre_counts = pd.Series([genre for sublist in imdb['Genres'] for genre in sublist])

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.countplot(data=genre_counts, y=genre_counts, color="#1C6AA6", order=genre_counts.value_counts().index)


# ### Distribution of ratings

# In[16]:


sns.countplot(data=imdb, x="Rating", color = '#1C6AA6')


# ### Top-rated French movies and their IMDb ratings 

# In[17]:


top_rated_french_movies = imdb.sort_values(by='Rating', ascending=False).head(10)

sns.set(style="whitegrid")  # Set the plot style
sns.barplot(data=top_rated_french_movies, x="Rating", y="Name", color="#1C6AA6")


# ### Distribution of release years

# In[18]:


plt.figure(figsize=(14, 7))

sort_years = imdb.sort_values(by='Year', ascending=True)

sns.countplot(data=sort_years, x="Year", color="#1C6AA6")
plt.xticks(rotation=90) 
plt.show()


# ### Year-wise Trends (Correlation between Year and, IMDB rating)

# In[19]:


year_rating = imdb.sort_values(by='Year', ascending=True)

plt.figure(figsize=(14,7))

sns.lineplot(data=year_rating, x="Year", y="Rating")
plt.xticks(rotation=90)
plt.show()


# ### Description Word Cloud Analysis

# In[20]:


# Concatenate all movie descriptions into a single string
all_descriptions = ' '.join(imdb['Desc'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

# Plot the word cloud
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ### Network Analysis - French movies that share common actors or directors

# In[21]:


# Create a network graph
G = nx.Graph()

# Process the data to add nodes and edges
for _, row in imdb.iterrows():
    # Split 'Director_Stars' into a list of directors and actors
    team = row['Director_Stars'].split(', ')
    movie_name = row['Name']
    
    # Add nodes (movies)
    G.add_node(movie_name)
    
    # Add edges between movies that share the same director or actor
    for i, row2 in imdb.iterrows():
        if row['Name'] != row2['Name']:
            other_team = row2['Director_Stars'].split(', ')
            common_members = set(team).intersection(set(other_team))
            if common_members:
                G.add_edge(movie_name, row2['Name'])

# Create an interactive network graph using Plotly
pos = nx.spring_layout(G, seed=42)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        colorbar=dict(thickness=15, title='Node Connections'),
        line_width=2))

node_trace.marker.color = [len(G.edges([node])) for node in G.nodes()]
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0)
                ))

fig.update_layout(
    autosize=False,
    width=1000,
    height=700,
)

# Show the interactive graph
fig.show()


# ## Sentimental Analysis 

# ### Lower case

# In[22]:


lower_case = imdb['Desc'].str.lower()


# In[23]:


lower_case


# ### Remove the punctuations

# In[24]:


# Review all the punctuations.
print(string.punctuation)


# In[25]:


cleaned_text = lower_case.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
print(cleaned_text)


# ### Tokenization

# In[26]:


tokenized_words = cleaned_text.str.split()

for words_list in tokenized_words:
    for word in words_list:
        print(f"'{word}'")


# ### Stop words

# In[27]:


stop_words = set()
with open("stopwords", "r") as file:
    for line in file:
        stop_words.add(line.strip())

filtered_words = []

for words_list in tokenized_words:
    for word in words_list:
        if word not in stop_words:
            filtered_words.append(word)

print(filtered_words)


# In[28]:


emotion_list = []
with open('emotions.txt', 'r') as file: 
    for line in file: 
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        #print("Word :" + word + " " + "Emotion :" + emotion)
        
        if word in filtered_words: 
            emotion_list.append(emotion)
print(emotion_list)


# In[29]:


w = Counter(emotion_list)
print(w)


# In[30]:


fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
ax1.set_xticklabels(w.keys(), rotation=90)
fig.tight_layout()
plt.savefig('graph.png')
plt.show()

