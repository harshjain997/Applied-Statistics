#!/usr/bin/env python
# coding: utf-8

# # Objective:
# We want to see if we can dive deep into this data to find some valuable insights for a given data set.

# # Domain: 
# Healthcare
# 
# Context :Leveraging customer information is paramount for most businesses. In the case
# of an insurance company, attributes of customers like the ones mentioned
# below can be crucial in making business decisions. Hence, knowing to explore
# and generate value out of such data can be an invaluable skill to have.

# # Import libraries

# In[1]:


import os #getting access to input files
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from sklearn.datasets import load_boston #for dealing with outliers
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import copy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir(r'C:\Users\harsh\Downloads')
print(os.getcwd())


# # Import the dataset

# In[3]:


df  = pd.read_csv('insurance.csv') # Import the dataset named 'Admission_predict.csv'

print(df.head())# view the first 5 rows of the data
print("---------------------------------------------------------------")
print(df.tail())#view the last 5 rows of the data


# # Shape and type of data

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


print(df.dtypes)


# In[7]:


df.dtypes.value_counts()


# # Check data set for further EDA , Finding missing values etc.

# In[8]:


df.isnull().sum() #faster way to check missing values but with info also we can find null values implementation is done above


# In[9]:


#alternate way for checking missing values column wise and taking out percentage of missing value in a particular data set

def missing_check(data):
    total = data.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (data.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(df)


# # Five number summary

# In[10]:


#this describe function gives five number summary for the given data and it also provide other stats for numerical data
#condition (data should be numerical int or float or boolean(1 binary digit which can store either 0 or 1))
df.describe().T


# for each column we got numerical column we got min,25%,50%,75%,max this conclude our five no. summary
# 75% of people have two or less children
# data looks good and age column also have resonable data that is distrbuted across

# # Distribution of  ‘bmi’, ‘age’ and ‘charges’ 

# In[11]:


sns.distplot(df['bmi'])


# Graph shows that the values are Uniformly distributed in bmi 

# In[12]:


sns.distplot(df['age'])


# Graph shows that the feature is non uniform in distribution

# In[13]:


sns.distplot(df['charges'])


# graph shows this feature is also non uniformly ditributed, as we can see it is right skewed and having mean greater than median

# # Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ 

# In[14]:


print("skewness for bmi",df["bmi"].skew())
print("skewness for age",df["age"].skew())
print("skewness for charges",df["charges"].skew())


# # Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

# In[15]:


plt.figure(figsize= (10,7))
plt.subplot(2,2,2)
sns.boxplot(x= df.bmi, color='lightblue')


# bmi has very less extreme values

# In[16]:


plt.figure(figsize= (10,7))
plt.subplot(2,2,2)
sns.boxplot(x= df.age, color='lightblue')


# as graph showing age doesn't have eextreme values 

# In[17]:


plt.figure(figsize= (10,7))
plt.subplot(2,2,2)
sns.boxplot(x= df.charges, color='lightblue')


# charges are highly skewed as shown above also and have so many extreme values

# # Distribution of categorical columns (include children) 

# In[18]:


plt.figure(figsize=(20,25))

x = df.smoker.value_counts().index    #Values for x-axis
y = [df['smoker'].value_counts()[i] for i in x]   # Count of each class on y-axis

plt.subplot(4,2,1)
plt.bar(x,y, align='center',alpha = 0.7)  #plot a bar chart
plt.xlabel('Smoker?')
plt.ylabel('Count ')
plt.title('Smoker distribution')


sns.stripplot(df['smoker'])


# we can see very less smoker as compared to non smokers

# In[19]:


plt.figure(figsize=(20,25))
x1 = df.sex.value_counts().index    #Values for x-axis
y1 = [df['sex'].value_counts()[j] for j in x1]   # Count of each class on y-axis

plt.subplot(4,2,2)
plt.bar(x1,y1, align='center',color = 'grey',alpha = 0.7)  #plot a bar chart
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender distribution')


# no significant difference between male and female smokers , almost equal 

# In[20]:


plt.figure(figsize=(20,25))
x2 = df.region.value_counts().index    #Values for x-axis
y2 = [df['region'].value_counts()[k] for k in x2]   # Count of each class on y-axis

plt.subplot(4,2,3)
plt.bar(x2,y2, align='center',alpha = 0.7)  #plot a bar chart
plt.xlabel('Region')
plt.ylabel('Count ')
plt.title("Regions' distribution")


# instances of smokers are distributed evenly across 4 regions , not much changes

# In[21]:


plt.figure(figsize=(20,25))
x3 = df.children.value_counts().index    #Values for x-axis
y3 = [df['children'].value_counts()[l] for l in x3]   # Count of each class on y-axis

plt.subplot(4,2,4)
plt.bar(x3,y3, align='center',color = 'lightgreen',alpha = 0.7)  #plot a bar chart
plt.xlabel('No. of children')
plt.ylabel('Count ')
plt.title("Children distribution")


# smoker have very less instance of 3,4,5 children

# # Pair plot that includes all the columns of the data frame 

# In[22]:


#sns.pairplot(df[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]) #this cannot give pair plot for categorical values 


df_encoded = copy.deepcopy(df)
df_encoded.loc[:,['sex', 'smoker', 'region']] = df_encoded.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) 

sns.pairplot(df_encoded)  #pairplot
plt.show()


# # Do charges of people who smoke differ significantly from the people who don't?

# In[23]:


print(df.smoker.value_counts())
sns.stripplot(df['charges'], df['smoker'])

plt.figure(figsize=(8,6))
sns.scatterplot(df.age, df.charges,hue=df.smoker,palette= ['orange','black'] ,alpha=0.6)
plt.show()


# In[24]:


# T-test to check dependency of smoking on charges
Ho = "Charges of smoker and non-smoker are same"   # Stating the Null Hypothesis
Ha = "Charges of smoker and non-smoker are not the same"   # Stating the Alternate Hypothesis

x = np.array(df[df.smoker == 'yes'].charges)  # Selecting charges corresponding to smokers as an array
y = np.array(df[df.smoker == 'no'].charges) # Selecting charges corresponding to non-smokers as an array

t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test

if p_value < 0.05:  # Setting our significance level at 5%
    print(f'{Ha} as the p_value ({p_value}) < 0.05')
else:
    print(f'{Ho} as the p_value ({p_value}) > 0.05')


# it means smokers seem to claim significantly more money than non-smokers

# # Does bmi of males differ significantly from that of females?

# In[25]:


print(df.sex.value_counts())
sns.stripplot(df['bmi'], df['sex'])

plt.figure(figsize=(8,6))
sns.scatterplot(df.age, df.charges,hue=df.sex,palette= ['red','grey'] )
plt.show()


# In[26]:


# T-test to check dependency of bmi on gender
Ho = "Gender has no effect on bmi"   # Stating the Null Hypothesis
Ha = "Gender has an effect on bmi"   # Stating the Alternate Hypothesis

x = np.array(df[df.sex == 'male'].bmi)  # Selecting bmi values corresponding to males as an array
y = np.array(df[df.sex == 'female'].bmi) # Selecting bmi values corresponding to females as an array

t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test

if p_value < 0.05:  # Setting our significance level at 5%
    print(f'{Ha} as the p_value ({p_value.round()}) < 0.05')
else:
    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')


# # Is the proportion of smokers significantly different in different genders?

# In[27]:


# Chi_square test to check if smoking habits are different for different genders
Ho = "Gender has no effect on smoking habits"   # Stating the Null Hypothesis
Ha = "Gender has an effect on smoking habits"   # Stating the Alternate Hypothesis

crosstab = pd.crosstab(df['sex'],df['smoker'])  # Contingency table of sex and smoker attributes

chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)

if p_value < 0.05:  # Setting our significance level at 5%
    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')
else:
    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')
crosstab


# Proportion of smokers in males is significantly different from that of the females we can see the proportion also as

# In[28]:


print("Total smoker count", df[df['smoker']=='yes'].shape[0]) 
print("Total male smokers", df[df['smoker']=='yes'][df['sex']=='male'].shape[0]) 
print("Total female smokers", df[df['smoker']=='yes'][df['sex']=='female'].shape[0]) 
print("Proportion of male smokers", (df[df['smoker']=='yes'][df['sex']=='male'].shape[0])/df[df['smoker']=='yes'].shape[0]) 
print("Proportion of female smokers", (df[df['smoker']=='yes'][df['sex']=='female'].shape[0])/df[df['smoker']=='yes'].shape[0])


# The proportions being 58% and 42% for male and female genders who smoke are not significantly different.

# # Is the distribution of bmi across women with no children, one child and two children, the same?# 

# In[29]:


sns.stripplot(df['bmi'], df[df['sex']=='female']['children'])


# Yes, the distributions of ‘bmi’ are nearly same across women with 0, 1 or 2 children.

# In[30]:


# Test to see if the distributions of bmi values for females having different number of children, are significantly different

Ho = "No. of children has no effect on bmi"   # Stating the Null Hypothesis
Ha = "No. of children has an effect on bmi"   # Stating the Alternate Hypothesis


female_df = copy.deepcopy(df[df['sex'] == 'female'])

zero = female_df[female_df.children == 0]['bmi']
one = female_df[female_df.children == 1]['bmi']
two = female_df[female_df.children == 2]['bmi']


f_stat, p_value = stats.f_oneway(zero,one,two)


if p_value < 0.05:  # Setting our significance level at 5%
    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')
else:
    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')


# In[ ]:




