## STREAMLIT DASHBOARD ##

#import packages/requirements
import selenium
import os,sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import json
from bs4 import BeautifulSoup
import numpy as np
import time
import streamlit as st 
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import dateparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus.reader.api import *
from nltk.corpus import opinion_lexicon
import matplotlib.pyplot as plt
from nltk import FreqDist
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk import ngrams, FreqDist
from bertopic import BERTopic
from umap import UMAP #import UMAP to get reproducable effects from bertopic
from sentence_transformers import SentenceTransformer, util
from hdbscan import HDBSCAN
import plotly.graph_objects as go
import seaborn as sns 
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

#st.sleep for timer?

#page layout
st.set_page_config(
     page_title='Restaurant Review Dashboard',
     layout="wide",
     initial_sidebar_state="expanded"#,
     #page_icon="IMG_1109.png"
     
)

#set sidebar
#st.sidebar.image("IMG_1109.png", use_container_width=True)
st.sidebar.title('Restaurant Review Dashboard')
st.sidebar.divider()
menu = st.sidebar.selectbox("Select Analysis Section", ["Home", "Word Analysis", "Topic Clustering", "Mentions", "What to Expect"])
st.sidebar.divider()
url = st.sidebar.text_input("Enter URL")

if menu == "Home":
    st.title("Home/How to Use!")

    #Intro
    
    st.write("Ever wonder what to order at a restaurant, but don't feel like reading the plethora of Google and Yelp reviews? Yea me too buddy. Hence why I created this *hopefully* cool and helpful dashboard!!")
    st.write("To ensure the best use-case of this dashboard, I will go through on how to set it up!")
    st.write("###### **DISCLAIMER: ** CHROME WILL OPEN UP AUTOMATICALLY AND AUTOMATICALLY SCRAPE. DONT BE SCARED IF IT POPS UP OUTTA NOWHERE")
    st.write("First up, you will input a Google Maps URL into the text box on the left hand side! Note that the URL has to be from a very specific page, such as this: ")
    st.image("example.png", use_container_width=True)
    st.write("To do this, you can click on the restaurant name and copy that URL! otherwise it will not run.")
    st.write("If you get the URL from a page that looks like the image down below, the code will **NOT RUN**!")
    st.image("bad_example.png",use_container_width=True)
    st.write("After inserting the URL, the code will run for a good while (like uhhh 2 minutes to maybe 10). You should see it scrape in real-time! (So cool right!!! Don't show all your excitement at once)")

    st.write("if you get an error similar to this: ")
    st.write("*'Unable to locate element: 'method':'xpath','selector':'//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[1]*")
    st.write("Refresh the page!")

@st.cache_data
####### PARSE REVIEWS#####
def scrape_data():

    chrome_options = Options()
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--no-sandbox") # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage") # overcome limited resource problems
    try: 
         service = Service(ChromeDriverManager().install())
         driver = webdriver.Chrome(service=service, options=chrome_options)
     
         driver.get(url)
     
         #may need to define xpath for "i agree" button. Did not pop up for me, will try on someone elses device later
         import time
         time.sleep(3)
     
         #obtain title
         rest_name = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[1]/h1').text
     
         #obtain resturant type
         rest_type = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[2]/div/div[2]/span[1]/span/button').text
     
         #obtain restaurant value
         value = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[2]/div/div[1]/span/span/span/span[2]/span/span').text
     
         #get restaurant address
         #address = driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div[8]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[9]/div[3]/button/div/div[2]/div[1]").text
         #can create a dictionary, key is state, div number is entry; search for state, if state, then that div number
     
         #if need to go thru newer reviews first, insert that in here:
     
         #code chunk below helps us find review button. Got this code off of medium
         driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Reviews')]").click()
     
         #obtain rating
         total_rating = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[1]').text
     
         #scroll till all reviews are loaded up
             #scroll by amount -- calculate and see how many reviews are in one scroll (10 scrolls is in one scroll)
         num_reviews = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[3]').text.split(" ")[0] #this code gives us the number
     
         #some reviews may have columns, will want to take that out
         if num_reviews.find(",") == True:
             num_reviews = num_reviews.replace(",","")
         else:
             num_reviews = num_reviews
     
         #now that we have number of reviews, we can scroll through reviews and load up each review
     
         height = 0
         while height <= (int(num_reviews)):
             try:
                 scroll_element = driver.find_element(By.XPATH, "//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]") #want to scroll first; finds scroll bar element
                 try: #find "more"
                     more_element = driver.find_element(By.XPATH, "//button[@aria-label='See more']")
                     if more_element.get_attribute("aria-expanded") == "false":
                         more_element.click()
                         time.sleep(.25) #might try (int(num_reviews)/10) (would need to divide by the number of 0's plus 2) len(str(num_reviews)).
                 except NoSuchElementException: #scroll if no "more"
                     driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_element)
                     time.sleep(.25)
             except NoSuchElementException:
                 print("Scrollbar element not found.")
                 break
             height += 1 #once height is reached... or it doesnt touch anymore, break
     
         #acquire reviews and parse them into a dataset #obtained from medium: https://medium.com/@isguzarsezgin/scraping-google-reviews-with-selenium-python-23135ffcc331
         reviews = BeautifulSoup(driver.page_source,'html.parser')
         driver.quit()
         return rest_name, rest_type, value, total_rating, num_reviews, reviews
    except Exception as e:
        st.error(f"Error scraping data: {e}")
        return None
     


if __name__ == "__main__":
    reviews = scrape_data()

    if reviews:
        rest_name, rest_type, value, total_rating, num_reviews, reviews = reviews

        st.sidebar.write(f"**Restaurant Name**: {rest_name}")
        st.sidebar.write(f"**Restaurant Type**: {rest_type}")
        st.sidebar.write(f"**Average Value Spend**: {value}")
        st.sidebar.write(f"**Average Rating (Google)**: {total_rating}")
        st.sidebar.write(f"**Number of Reviews**: {num_reviews}")

#Extract restaurant basic informaton: Name, Address, Number of Reviews, Average Rating, Restaurant Type, Map
# st.sidebar.write(f"**Restaurant Name**: {rest_name}")
# st.sidebar.write(f"**Restaurant Type**: {rest_type}")
# st.sidebar.write(f"**Average Value Spend**: {value}")
# st.sidebar.write(f"**Average Rating (Google)**: {total_rating}")
# st.sidebar.write(f"**Number of Reviews**: {num_reviews}")
#print(address)

########### Create Datasets #############
import pandas as pd

def get_reviews(reviews):

    #create a dictionary to store reviews in
    review_dict = {
        "Review Rating":[],
        "Review Time": [],
        "Review Text": []
    }
    #Write for loop to gather information
    for result in reviews.find_all('div', class_='jJc9Ad'): 
        review_text_element = result.find('span', class_='wiI7pd')
        if review_text_element:  # Only process reviews with text (earlier got mismatch array sizes)
            # Extract review text
            review_text = review_text_element.text
            # Extract review rating
            review_rating_element = result.find('span', class_='kvMYJc')
            review_rating = review_rating_element["aria-label"]
            # Extract review time
            review_time_element = result.find('span', class_='rsqaWe')
            review_time = review_time_element.text
            # Append data to the dictionary
            review_dict["Review Rating"].append(review_rating)
            review_dict["Review Time"].append(review_time)
            review_dict["Review Text"].append(review_text)

    return(pd.DataFrame(review_dict)) 

#get the top words used in the reviews
def get_top_mentions(reviews):
    mentions_dict = {
        "Mention": []
    }

    for result in reviews.find_all('div', class_='KNfEk aUjao'): 

        # Extract mention
        mention_element = result.find('button', class_="e2moi")
        mention = mention_element["aria-label"]
        mentions_dict["Mention"].append(mention)

    return pd.DataFrame(mentions_dict)


def get_other_stuff(reviews):

    stuff = []

    for result in reviews.find_all('span', class_="RfDO5c"):

        stuff_dict = {
                "Service_Type": None,
                "Meal Type": None,
                "Price per Person": None,
                "Food": None,
                "Service": None,
                'Atmosphere': None,
                'Recommended Dishes': None,
                'Parking Space': None,
                'Parking Options': None,
                'Wheelchair Accessible': None
            }
        
        #get service
        service_locate = result.find(string="Service")
        service = service_locate.find_next('span').get_text() if service_locate else None
        stuff_dict["Service_Type"] = service

        #get meal type
        meal_type_locate = result.find(string="Meal type")
        meal_type = meal_type_locate.find_next('span').get_text() if meal_type_locate else None
        stuff_dict["Meal Type"] = meal_type

        #get price per person
        pp_locate = result.find('span', attrs={'aria-label': True})
        pp = pp_locate['aria-label'] if pp_locate else None
        stuff_dict["Price per Person"] = pp

        # Extract ratings for Food, Service, Atmosphere, etc.
        ratings = {}
        for tag in result.find_all('b'):
            key = tag.get_text(strip=True).replace(":", "")  # Remove colon
            value = tag.next_sibling.strip() if tag.next_sibling else None
            if value and value.isdigit():
                ratings[key] = int(value)
            else:
                ratings[key] = value

        # Assign ratings to corresponding keys in the dictionary
        stuff_dict["Food"] = ratings.get("Food", None)
        stuff_dict["Service"] = ratings.get("Service", None)
        stuff_dict["Atmosphere"] = ratings.get("Atmosphere", None)

        #get recommended dishes
        recommended_locate = result.find(string = 'Recommended dishes')
        recommend = recommended_locate.find_next('span').get_text() if recommended_locate else None
        stuff_dict["Recommended Dishes"] = recommend

        #get parking space
        parking_locate = result.find(string="Parking space")
        parking_space = parking_locate.find_next('span').get_text() if parking_locate else None
        stuff_dict["Parking Space"] = parking_space

        #get parking options
        parking_locate = result.find(string="Parking options")
        parking_options = parking_locate.find_next('span').get_text() if parking_locate else None
        stuff_dict["Parking Options"] = parking_options

        #Wheelchair accessibility
        wheelchair_locate =result.find(string="Wheelchair accessibility")
        wheelchair = wheelchair_locate.find_next('span').get_text() if wheelchair_locate else None
        stuff_dict['Wheelchair Accessible'] = wheelchair

        stuff.append(stuff_dict)

    # Convert to DataFrame
    return pd.DataFrame(stuff)

#create datasets
reviews_set = get_reviews(reviews)
mentions_set = get_top_mentions(reviews)
stuff_set = get_other_stuff(reviews) #the way the data was obtained/exported, grab counts/means with na exclusions


######### Text Analytics ##########

## First Attempt -- Sentiment didnt match reviews well ##

#clean dataset
# get rid of the word "stars" and get numbers
reviews_set["Review Rating"] = reviews_set["Review Rating"].apply(lambda x: x.split(" ")[0])
reviews_set["Review Rating"] = reviews_set["Review Rating"].apply(int)

#get specific time periods/dates
reviews_set["Review Time"] = reviews_set["Review Time"].apply(dateparser.parse)
reviews_set["Review Time"] = reviews_set["Review Time"].apply(lambda x: x.strftime("%Y-%m-%d"))

#clean up review text

# Define stop words and punctuation
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

def clean_text(sentence):
    words_wout_punc = re.sub(r"[^\w\s']", "", sentence) #remove punctuation
    token_words = word_tokenize(words_wout_punc.lower()) #turn sentences to lowercase
    token_words = [word for word in token_words if word not in stop_words] #removes stopwords/unimportant words
    lemmatizer = WordNetLemmatizer() #initialize the lemmatizer thingy
    lemmatized_words = [lemmatizer.lemmatize(word) for word in token_words] #makes words uniform across the board
    return ' '.join(lemmatized_words) #join back the words and return them for cleaned reviews
    
reviews_set["Clean Reviews"] = reviews_set["Review Text"].apply(clean_text)

def clean_text_wout_stop(sentence): #doing this to get more fair setniment in reviews
    words_wout_punc = re.sub(r"[^\w\s']", "", sentence) #remove punctuation
    token_words = word_tokenize(words_wout_punc.lower()) #turn sentences to lowercase
    lemmatizer = WordNetLemmatizer() #initialize the lemmatizer thingy
    lemmatized_words = [lemmatizer.lemmatize(word) for word in token_words] #makes words uniform across the board
    return ' '.join(lemmatized_words) #join back the words and return them for cleaned reviews
    
reviews_set["Clean Reviews Stop"] = reviews_set["Review Text"].apply(clean_text_wout_stop)

#ok now for the good stuff, the text analytics portion. Lets get into it (this chunk is sentiment)
def sentiment_index(text):
    sia = SentimentIntensityAnalyzer() #initialize the sentiment analyzer
    sentiment_index = sia.polarity_scores(text) #get sentiment scores
    return sentiment_index

reviews_set[["neg","neu","pos","compound"]] = reviews_set["Clean Reviews Stop"].apply(sentiment_index).apply(pd.Series)
#do a better split of sentiment analysis


## Second Attempt -- Matches better ##


# Download the VADER lexicon
nltk.download('vader_lexicon')
# Lists to store sentiment scores
positive_scores = []
negative_scores = []
neutral_scores = []
sia = SentimentIntensityAnalyzer()
for review in reviews_set["Clean Reviews Stop"]:
    # Get sentiment scores for the review
    sentiment = sia.polarity_scores(review)
    # Store the individual sentiment scores in lists
    positive_scores.append(sentiment['pos'])
    negative_scores.append(sentiment['neg'])
    neutral_scores.append(sentiment['neu'])
reviews_set['negative_score'] = negative_scores
reviews_set['neutral_score'] = neutral_scores
reviews_set['positive_score'] = positive_scores
#### Count Number of Positive, Negative, and Neutral terms ####
# Ensure you have the lexicon downloaded
nltk.download('opinion_lexicon')
nltk.download('punkt')
# Lists of positive and negative words
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())
negative_counts = []
neutral_counts = []
positive_counts = []
positive_percentage = []
negative_percentage = []
review_class = []
# Iterate through each review using iterrows
for index, row in reviews_set.iterrows():
    review = row["Clean Reviews Stop"]
    rating = row['Review Rating']
    # Tokenize review into words
    words = word_tokenize(review.lower())
    # Count positive, negative, and neutral terms
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    neu_count = len(words) - pos_count - neg_count if len(words) > 0 else 0  # Handle empty reviews
    negative_counts.append(neg_count)
    neutral_counts.append(neu_count)
    positive_counts.append(pos_count)
    # Calculate positive percentage
    if (pos_count + neg_count) > 0:
        positive_percentage.append(pos_count / (pos_count + neg_count) * 100)
        negative_percentage.append(neg_count / (pos_count + neg_count) * 100)
    else:
        positive_percentage.append(0)
        negative_percentage.append(0)
    # Determine review class using the current rating
    if pos_count > neg_count and rating != 1:
        review_class.append('positive')
    elif neg_count > pos_count and rating != 5:
        review_class.append('negative')
    elif rating == 1:
        review_class.append('negative')
    elif rating == 5:
        review_class.append('positive')
    else:
        review_class.append('neutral')
# Add counts and percentages to DataFrame
reviews_set['negative_count'] = negative_counts
reviews_set['neutral_count'] = neutral_counts
reviews_set['positive_count'] = positive_counts
reviews_set['percentage_positive'] = positive_percentage
reviews_set['percentage_negative'] = negative_percentage
reviews_set['review_class'] = review_class
# Convert review_class to numerical values
reviews_set['review_num_class'] = 0  # Default to 0 for neutral
reviews_set.loc[reviews_set['review_class'] == 'positive', 'review_num_class'] = 1
reviews_set.loc[reviews_set['review_class'] == 'negative', 'review_num_class'] = -1


####### Word Analysis #####
if menu == "Word Analysis":
    st.title("Word Analysis")
    st.write(f"Self-Explanatory, but this section tells you the most common words at {rest_name}! This enables you to kinda gauge what people talk about and order!")
    #get top words

    # all_words_frequency = []

    # for review in reviews_set["Clean Reviews"]:
    #     words = review.split(sep=" ") 
    #     all_words_frequency.extend(words) 
    # fq = FreqDist(all_words_frequency)

    # display(fq.most_common(30))
    # fq.plot(30)

    # ... (your data loading and processing for reviews_set)

    all_words_frequency = []
    for review in reviews_set["Clean Reviews"]:
        words = review.split(sep=" ")
        all_words_frequency.extend(words)
    fq = FreqDist(all_words_frequency)

    sns.set_theme(style="white", font="Times New Roman")

    top_words = fq.most_common(30)
    words, counts = zip(*top_words)  # Unzip the tuples into separate lists

    fig, ax = plt.subplots()
    fig.set_facecolor('#f2f0ef')
    sns.lineplot(x=list(words), y=list(counts), ax=ax, color = '#abcda1')  # Use Seaborn's barplot
    ax.set_facecolor('#f2f0ef')
    ax.set_title(f'Top 30 Most Frequent Words at {rest_name}')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=75)
    for i in ax.containers:
        ax.bar_label(i)

    plt.tight_layout()

    st.write(f"This chart below tells you about the most common words in the reviews at {rest_name}")
    st.pyplot(fig)



    #get common words that come together -- want to graph this

    all_counts = dict()
    for size in 2, 3, 4, 5:
        all_counts[size] = FreqDist(ngrams(all_words_frequency, size))

    #can clean this all up with a loop -- do that soon buddy

    top_words_2 = all_counts[2].most_common(20)
    top_words_3 = all_counts[3].most_common(20)
    top_words_4 = all_counts[4].most_common(20)


    top_words_2 = pd.DataFrame(top_words_2)
    top_words_3 = pd.DataFrame(top_words_3)
    top_words_4 = pd.DataFrame(top_words_4)


    top_word_combos = pd.concat([top_words_2,top_words_3,top_words_4])

    top_word_combos[0] = top_word_combos[0].apply(lambda x: ' '.join(x))

    top_word_combos = top_word_combos.sort_values(1, axis=0, ascending=False).head(20)

    fig, ax = plt.subplots()
    fig.set_facecolor('#f2f0ef')
    sns.barplot(x=list(top_word_combos[1]), y=list(top_word_combos[0]), ax=ax, color = '#abcda1',orient='h')  # Use Seaborn's barplot
    ax.set_facecolor('#f2f0ef')
    ax.set_title(f'Top 20 Most Frequent Word Pairs at {rest_name}')
    ax.set_ylabel('Word Pairs')
    ax.set_xlabel('Frequency')
    ax.tick_params(axis='x')
    for i in ax.containers:
        ax.bar_label(i)

    st.write(f"The chart below will tell you the most frequent word pairs seen in the reviews!")

    st.pyplot(fig)


######## TOPIC CLUSTERING ###########
if menu == "Topic Clustering":
    st.title("Topic Clustering")


    st.write("Now this page may be confusing, but don't fret, since **I** am here! (ok maybe fret a little)")
    st.write("The *interactive* chart below will allow you to see the topics that people write about! (cool right 😎). The bigger the circle, the more frequent the topic pops up!")
    st.write("The circles inside of other circles kinda tell us what topics stem from the bigger topic. My hope for this analysis is to showcase what people talk about in the reviews mainly!")

    #ok now for the good stuff, the text analytics portion. Lets get into it (this chunk is TOPICS)
    #Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(n_neighbors=2, n_components=4, min_dist=0.0, metric='cosine')  #sets seed for reproducable effects

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=4, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # All steps together
    topic_model = BERTopic(
    embedding_model=embedding_model,    # Step 1 - Extract embeddings
    umap_model=umap_model,              # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
    calculate_probabilities=True,
    verbose=True,
    )

    topics, probs = topic_model.fit_transform(reviews_set["Clean Reviews"])
    #topic_model.get_topic_info()

    fig = topic_model.visualize_topics()

    st.plotly_chart(fig)

    st.write("The visualization below shows you the top words within the topics, and also enables you to create a potential summary of the topic based on what you see!")

    #get a list of topics and choose based on that list?
    fig = topic_model.visualize_barchart(top_n_topics=8,n_words=7,title = f'Topical Words at {rest_name}')
    st.plotly_chart(fig)


########## MENTIONS #########
if menu == "Mentions":
    st.title("Mentions")
    st.write("I did this part because I was personally TIRED of reading all the reviews for recommended dishes and tallying them (I did not do this, but like I would have!)")


    #make it so mentions and number are on 2 different columns
    mentions_set["Counts"] = mentions_set["Mention"].apply(lambda x: x.split(", ")[1])
    mentions_set["Mention"] = mentions_set["Mention"].apply(lambda x: x.split(", ")[0])

    mentions_set["Counts"]= mentions_set["Counts"].apply(lambda x: x.split(" ")[2])
    mentions_set["Counts"]= mentions_set["Counts"].apply(int)

    #create a barchart for visualization
    sns.set_theme(style="white",font="Times New Roman")

    fig, ax = plt.subplots()
    fig.set_facecolor('#f2f0ef') #faf0e6

    ax = sns.barplot(x= mentions_set["Counts"], y = mentions_set["Mention"],color = '#abcda1',orient='h') #maybe want to do it so color is chosen basd on the restaurant vibe?

    ax.set_fc('#f2f0ef')

    plt.ylabel('Top Google Mentions')
    plt.xlabel('Count of Google Mentions')
    plt.title(f'Top Mentions from Reviews at {rest_name}') 

    for i in ax.containers:
        ax.bar_label(i,)

    st.write(f"The chart below will tell you the top mentions @ {rest_name} that Google shows you itself! (No need for me here, but it is good to have 🫡)")
        
    st.pyplot(fig)

    # stuff_set["Service_Type"].value_counts()

    # for i in stuff_set.columns:
    #    print(stuff_set[i].value_counts())


    rec_dishes = []
    for i in stuff_set["Recommended Dishes"]:
        if i is not None:
            dish = i.split(", ")
            rec_dishes.append(dish)
        else:
            continue

    ## CAN DO ASSOCIATION ANALYSIS HOLY SMOKES (on rec_dishes)

    #work with recommended dishes
    order = pd.DataFrame(rec_dishes)
    for i in range(len(order.columns)):
        if i == 0:
            dishes = pd.concat([order[i],order[i+1]]).reset_index(drop=True)
        else:
            dishes = pd.concat([dishes,order[i]]).reset_index(drop=True)

    #turn blanks into nulls
    dishes = np.where(dishes == "",np.nan,dishes)
    #remove rows will nulls --> dataset is now cleaned #yayuh
    dishes = pd.DataFrame(dishes)
    dishes = dishes.dropna(axis=0)
    #after, get count of dishes and proportion of those dishes out of the other dishes
    recs = dishes.value_counts().reset_index()
    ## Get proportions of each and make charts?



    #graph top dishes

    sns.set_theme(style="white",font="Times New Roman")

    fig, ax = plt.subplots()
    fig.set_facecolor('#f2f0ef') #faf0e6

    ax = sns.barplot(x= recs['count'], y = recs[0],color = '#abcda1',orient='h') #maybe want to do it so color is chosen basd on the restaurant vibe?

    ax.set_fc('#f2f0ef')

    plt.ylabel('Scraped Dishes')
    plt.xlabel('Count of Dishes')
    plt.title(f'Top Dishes at {rest_name}') 

    #top dish?
    # st.sidebar.write(f'Top Dish: {}')


    for i in ax.containers:
        ax.bar_label(i,)
        
    st.write(f"The chart below will tell you the top mentions @ {rest_name} that I scraped! Huzzah! All those hours and hours of tallying now only takes a way shorter amount of time! Definitely use this chart to see what people order and go based off of that!")

    st.pyplot(fig)

    st.write("Now, as a foodie, I cannot choose *just one* dish. I definitely need to try more! The table down below will tell you the most commonly ordered dishes thru an association analysis!")

    #association analysis
    #add index to make wide to long
    order['index'] = range(1,len(order)+1)

    move_index = order.pop('index')

    order.insert(0,'index',move_index)

    #order_long = pd.wide_to_long(order,'orders',i='index',j='order')

    order_long = order.melt(id_vars=['index'], 
                        value_vars= order.iloc[:,range(1,len(order.columns))],
                        var_name='Meal_Number', 
                        value_name='Food')

    #make transactional -- maybe remove nulls?
    ordered_transactions = order_long.groupby('index')['Food'].apply(list).reset_index()

    ordered_transactions_binary = pd.crosstab(order_long['index'],order_long['Food']).astype('bool').astype('int')


    frequent_itemsets = apriori(ordered_transactions_binary, min_support=0.03, use_colnames=True) #get min_support based on some math, so its good for each place

    rules = association_rules(frequent_itemsets, metric="lift")

    rules = rules.sort_values('lift',ascending=False)

    #gorup by lift and leverage and remove duplicates that way
    # order by confidence
    rules = rules.sort_values('confidence', ascending= False)
    pairings = rules[['antecedents','consequents']].head(20)
    st.write(pairings)

if menu == "What to Expect":
    st.title("What to Expect")

    #data for graphs below
    service_counts = stuff_set["Service_Type"].value_counts()
    service_counts = pd.DataFrame(service_counts).reset_index()

    meal_counts = stuff_set["Meal Type"].value_counts()
    meal_counts = pd.DataFrame(meal_counts).reset_index()

    price_counts = stuff_set["Price per Person"].value_counts()
    price_counts = pd.DataFrame(price_counts).reset_index()
    price_counts['Price per Person'] = price_counts['Price per Person'].astype(str)  
    prices_old = ['$1 to $10', '$10 to $20', '$20 to $30', '$30 to $50', '$50 to $100', '$100 or Above']
    prices_new = ['$1 - 10', '$10 - 20', '$20 - 30', '$30 - 50', '$50 - 100', '$100 or Above']
    price_counts['Price per Person'] = np.select([price_counts['Price per Person'] == old for old in prices_old], prices_new, default=price_counts['Price per Person'])

    parking_space = stuff_set["Parking Space"].value_counts()
    parking_space = pd.DataFrame(parking_space).reset_index()

    parking_options = stuff_set["Parking Options"].value_counts()
    parking_options = pd.DataFrame(parking_options).reset_index()

    wheelchair_accessible = stuff_set["Wheelchair Accessible"].value_counts()
    wheelchair_accessible = pd.DataFrame(wheelchair_accessible).reset_index()


    #combine all stuff_set graphs into one chunk of code and plot out in an array -- wheelchair accessible is going to be seperate, since it varies

    sns.set_theme(style="white", font="Times New Roman")

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    fig.set_facecolor('#f2f0ef')  # faf0e6

    colors = ['#abcda1', '#A1CDC3','#A1C1CD', '#A1ABCD', '#ADA1CD','#CDA1AB','#CDB1A1','#CDC3A1']

    # list of dataframes to scope through
    list_dfs = [service_counts, meal_counts, price_counts, parking_space, parking_options]
    titles = ['Service Types', 'Meal Types', 'Price per Person', 'Parking Space', 'Parking Options']

    for data, ax, titles in zip(list_dfs, axes.flatten(), titles):
        wedges, texts = ax.pie(data["count"], labels=data.iloc[:, 0],
                                        colors=colors, labeldistance=1.05,
                                        wedgeprops={'linewidth': 2, 'edgecolor': '#f2f0ef'})
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.7, fc='#f2f0ef')
        ax.add_artist(centre_circle)
        ax.set_title(f'{titles} at {rest_name}',fontweight='bold', fontsize = 14)
        #ax.legend(title = f'{titles} at {rest_name}',bbox_to_anchor=(1.25, -.35), loc='lower center')

    #remove other subplot
    axes[2, 1].remove()

    plt.tight_layout()

    st.write(f"To prepare your visit to {rest_name}, I took the time to show you what other customers say in terms of spending, parking, and the vibe people go for the most!")

    st.pyplot(fig)

    sns.set_theme(style="white",font="Times New Roman")

    fig, ax = plt.subplots()

    fig.set_facecolor('#f2f0ef') #faf0e6

    colors = ['#abcda1', '#A1CDC3','#A1C1CD', '#A1ABCD', '#ADA1CD','#CDA1AB','#CDB1A1','#CDC3A1']

    ax = plt.pie(wheelchair_accessible["count"], labels = wheelchair_accessible["Wheelchair Accessible"], 
                colors = colors, labeldistance= 1.05,
                wedgeprops = { 'linewidth' : 2, 'edgecolor' : '#f2f0ef' })

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='#f2f0ef')
    circ = plt.gcf()
    
    # Adding Circle in Pie chart
    circ.gca().add_artist(centre_circle)

    plt.title(f'Wheelchair Accessiblity at {rest_name}', loc = 'center',fontweight='bold', fontsize = 14) #can maybe add a snippet saying f"based on our anaylsis, we recommend you {top type}"
    #plt.legend(title = f'Wheelchair Accessiblity at {rest_name}',bbox_to_anchor=(.5, -.35), loc='lower center')
    plt.show()
    st.pyplot(fig)
