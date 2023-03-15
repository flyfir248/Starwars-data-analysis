import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

df=pd.read_csv("star_wars.csv")

print(df)

msno.matrix(df)

# save the plot as a PNG file
plt.savefig('missingness.png')


# Data cleaning
df = df.dropna()  # Drop rows with missing values
df = df.rename(columns={'Have you seen any of the 6 films in the Star Wars franchise?': 'seen_any'})
df = df.drop(columns=['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 28'])
df['Ranking'] = pd.to_numeric(df['Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.'])

# Descriptive analysis
fig = plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Age', hue='Gender', multiple='stack')
plt.title('Distribution of Respondents by Age and Gender')
fig.savefig('age_gender_distribution.png')

fig = plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='seen_any', hue='Gender')
plt.title('Have You Seen Any Star Wars Films?')
fig.savefig('seen_any.png')

# Exploratory data analysis
fig = plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Household Income', y='Ranking')
plt.title('Ranking of Star Wars Films by Household Income')
fig.savefig('ranking_by_income.png')

fig = plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Ranking')
plt.title('Ranking of Star Wars Films by Age')
fig.savefig('ranking_by_age.png')

# Hypothesis testing
male_ranks = df[df['Gender'] == 'Male']['Ranking']
female_ranks = df[df['Gender'] == 'Female']['Ranking']
t, p = ttest_ind(male_ranks, female_ranks)
print('t-test results: t = {:.2f}, p = {:.4f}'.format(t, p))

# Text analysis
from textblob import TextBlob

comments = df['Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.'].dropna()

polarity = []
subjectivity = []
for comment in comments:
    blob = TextBlob(comment)
    polarity.append(blob.sentiment.polarity)
    subjectivity.append(blob.sentiment.subjectivity)

fig = plt.figure(figsize=(8, 6))
sns.scatterplot(x=subjectivity, y=polarity)
plt.title('Sentiment Analysis of Star Wars Characters')
plt.xlabel('Subjectivity')
fig.savefig('sentiment_analysis.png')