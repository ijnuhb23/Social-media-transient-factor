
# imports needed
import pandas as pd
from pmaw import PushshiftAPI
import datetime as dt

api = PushshiftAPI()

# timeframe of extraction
before = int(dt.datetime(2022,5,1,0,0).timestamp())
after = int(dt.datetime(2010,4,1,0,0).timestamp())

# subreddit to perform the search
subreddit="wallstreetbets"

#limit of comments scrapred
limit=1000000

stock_list = ['TSLA'
]


# place holder for big stocks : 'AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU'
# place holder for other stocks : 'AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
## loop to scrape all stocks above
for i in stock_list:
    j = str(i)
    print(j)
# Stock name to be scraped
    q = [j]
    stock_name = j

# scrape comments
    comments = api.search_comments(q=q,subreddit=subreddit, limit=limit, before=before, after=after)

# indicator showing how many comments have been scraped
    print(f'Retrieved {len(comments)} comments from Pushshift')

# create a dataframe for comments
    comments_df = pd.DataFrame(comments)

# preview the comments data
# comments_df.head(50)

# export the comments results to csv
    comments_df.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_wsb_scraped_comments//'+str(stock_name)+'_wsb_comments.csv', header=True, index=False, columns=list(comments_df.axes[1]))