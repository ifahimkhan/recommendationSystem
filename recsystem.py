#!/usr/bin/env python
# coding: utf-8

# **Business Problem**
# 
# MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.      
# The dataset can be downloaded from here  -- (https://grouplens.org/datasets/movielens/100k/)
# This data set consists of: 
# 	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
# 	* Each user has rated at least 20 movies. 
#     * Simple demographic info for the users (age, gender, occupation, zip)
# 
# The data was collected through the MovieLens web site (movielens.umn.edu) during the seven-month period from September 19th,1997 through April 22nd, 1998.

# **Task and Approach:**
# 
# We need to work on the MovieLens dataset and build a model to recommend movies to the end users

# **Step 1 :** Importing Libraries and Understanding Data

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# To make data visualisations display in Jupyter Notebooks 
import numpy as np   # linear algebra
import pandas as pd  # Data processing, Input & Output load
#import matplotlib.pyplot as plt # Visuvalization & plotting
#import seaborn as sns # Also for Data visuvalization 

from sklearn.metrics.pairwise import cosine_similarity  # Compute cosine similarity between samples in X and Y.
from scipy import sparse  #  sparse matrix package for numeric data.
from scipy.sparse.linalg import svds # svd algorithm

import warnings   # To avoid warning messages in the code run
warnings.filterwarnings("ignore")


# **Step 2 :** Loading Data  & Corss chekcing 

# In[2]:


import os

os.chdir(r"C:\USB Fahim\Desktop\Python\FlaskProject")


# In[3]:


Rating = pd.read_csv('Ratings.csv') 
Movie_D = pd.read_csv('Movie details.csv',encoding='latin-1') ##Movie details 
User_Info = pd.read_csv('user level info.csv',encoding='latin-1') ## if you have a unicode string, you can use encode to convert


# In[4]:


Movie_D.head()


# In[5]:


# In[7]:


Rating.head()


# * Item id means it is Movie id 
# * Item_ID chnaged as Movie id for the better redability pupose 
# 

# In[8]:


Rating.columns = ['user_id', 'movie_id', 'rating', 'timestamp'] 


# Renaming the columns to avoid the space in the column name text 

# In[9]:


Movie_D.shape


# In[10]:


Movie_D.head()


# In[11]:


Movie_D.columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date ',
       'IMDb_URL', 'unknown', 'Action ', 'Adventure', 'Animation',
       'Childrens', 'Comedy ', 'Crime ', ' Documentary ', 'Drama',
       ' Fantasy', 'Film-Noir ', 'Horror ', 'Musical', 'Mystery',
       ' Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


# Renaming the columns to avoid the space in the column name text 

# **To get our desired information in a single dataframe, we can merge the two dataframes objects on the movie_Id column since it is common between the two dataframes.**
# 
# **We can do this using merge() function from the Pandas library**

# In[12]:


Movie_Rating = pd.merge(Rating ,Movie_D,on = 'movie_id')
Movie_Rating.describe()


# **We can see the Average rating for all the movie is 3.5**              
# **We can also see 25 percentile also indicating avaerage is 3 highest is 5**

# In[13]:


n_users = Movie_Rating.user_id.unique().shape[0]
n_items = Movie_Rating.movie_id.unique().shape[0]
print(n_items,n_users)


# No of unique users & No of unique Movies 

# In[14]:


# Calculate mean rating of all movies 
Movie_Stats = pd.DataFrame(Movie_Rating.groupby('movie_title')['rating'].mean())
Movie_Stats.sort_values(by = ['rating'],ascending=False).head()


# **Let's now plot the total number of ratings for a movie**

# In[15]:


# Calculate count rating of all movies 

Movie_Stats['Count_of_ratings'] = pd.DataFrame(Movie_Rating.groupby('movie_title')['rating'].count())
Movie_Stats.sort_values(by =['Count_of_ratings'], ascending=False).head()


# **Now we know that both the average rating per movie and the number of ratings per movie are important attributes**

# **Plot a histogram for the number of ratings**

# In[16]:


#Movie_Stats['Count_of_ratings'].hist(bins=50)


# **From the output, you can see that most of the movies have received less than 50 ratings.**
# It is evident that the data has a weak normal distribution with the mean of around 3.5. There are a few outliers in the data

# In[17]:


#sns.jointplot(x='rating', y='Count_of_ratings', data=Movie_Stats)


# * The graph shows that, in general, movies with higher average ratings actually have more number of ratings, compared with movies that have lower average ratings.

#  ### Finding Similarities Between Movies

# * We will use the correlation between the ratings of a movie as the similarity metric.
# * To see the corrilation we will create Pivot table between user_id ,movies, ratings

# In[18]:


User_movie_Rating = Movie_Rating.pivot_table(index='user_id', columns='movie_title', values='rating')
User_movie_Rating.head()


# In[19]:


##We can achieve this by computing the correlation between these two movies ratings and the ratings of the rest of the movies in the dataset. 
##The first step is to create a dataframe with the ratings of these movies 

# Example pick up one movie related rating  
User_movie_Rating['Air Force One (1997)']


# ## Correlation Similarity

# * We can find the correlation between the user ratings for the **given movie**  and all the other movies using corrwith() function as shown below:

# In[20]:


Similarity = User_movie_Rating.corrwith(User_movie_Rating['Air Force One (1997)'])
Similarity.head()


# In[21]:


corr_similar = pd.DataFrame(Similarity, columns=['Correlation'])
corr_similar.sort_values(['Correlation'], ascending= False).head(10)


# #### We will add the count of rating also to see why many movies are exactly correlating for the single movie 

# In[22]:


corr_similar_num_of_rating = corr_similar.join(Movie_Stats['Count_of_ratings'])
corr_similar_num_of_rating.sort_values(['Correlation'], ascending= False).head(10)


# * We can able to see  that a movie cannot be declared similar to the another movie based on just 2 or 3  ratings. 
# 
# * This is why we need to filter  movies correlated to given movie  that have more than 30/50 ratings

# In[23]:


corr_similar_num_of_rating[corr_similar_num_of_rating ['Count_of_ratings']>50].sort_values('Correlation', ascending=False).head()


# **Creation the user defined function to get the similar movies to recommend**
# * All the above steps created as one UDF so that we can pass the movie title and get the recomendations
# 

# In[24]:


def get_recommendations(title):
    # Get the movie ratings of the movie that matches the title
    Movie_rating = User_movie_Rating[title]

    # Get the  similarity corrilated  scores of all movies with that movie
    sim_scores = User_movie_Rating.corrwith(Movie_rating)

    # Sort the movies based on the similarity scores
    corr_title = pd.DataFrame(sim_scores, columns=['Correlation'])
    
    # Removing na values 
    corr_title.dropna(inplace=True)
    
    corr_title = corr_title.join(Movie_Stats['Count_of_ratings'])
    
    # Return the top 10 most similar movies
    return corr_title[corr_title ['Count_of_ratings']>50].sort_values('Correlation', ascending=False).head()


# In[27]:
def get_Suggetion():

    listOfMovies=Movie_D['movie_title']
    return listOfMovies
    

# Usage of the above function
get_recommendations('Black Sheep (1996)')


# In[29]:


get_recommendations('Star Wars (1977)')


