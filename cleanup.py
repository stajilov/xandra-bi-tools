import pandas as pd

df = pd.read_csv('tweets.csv', sep='\t', header=None)
print(df.head(5))
print(df.columns)
df = df.rename(columns={0:'sentiment', 1:'text'})

print(df.head(5))

df['url'] = 'https://bbc.com'
df['category'] = 'Science'
df['title'] = 'Lorem Title Here'

df.to_csv('clean_tweets.csv', index=False)
