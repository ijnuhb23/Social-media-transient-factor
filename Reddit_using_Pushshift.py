
# imports needed
import pandas as pd
from pmaw import PushshiftAPI
import datetime as dt

api = PushshiftAPI()

# timeframe of extraction
before = int(dt.datetime(2022,5,1,0,0).timestamp())
after = int(dt.datetime(2010,4,1,0,0).timestamp())

# term to be scraped (could be done by sector)
q = ['SBUX']

# subreddit to perform the search
subreddit="wallstreetbets"

#limit of comments scrapred
limit=500000

# scrape comments
comments = api.search_comments(q=q,subreddit=subreddit, limit=limit, before=before, after=after)

# indicator showing how many comments have been scraped
print(f'Retrieved {len(comments)} comments from Pushshift')

# create a dataframe for comments
comments_df = pd.DataFrame(comments)

# preview the comments data
comments_df.head(50)

# export the comments results to csv
comments_df.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//wsb_comments_SBUX.csv', header=True, index=False, columns=list(comments_df.axes[1]))