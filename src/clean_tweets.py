# --- Import libraries ---
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from re import match


# --- Functions ---
def clean_each_tweet(tweet_as_list_of_words, return_type=None):
    """
    Process each individual tweet (called by function clean_tweets)

    Arguments
    ---------
    :param tweet_as_list_of_words:  a tweet as a list of words

    Optional Arguments
    ------------------
    :param return_type:             type of processed tweet to return
                                    default to None
                                    can also choose ["stemmed", "lemmed"]

    Return
    ------
    :return:                        processed tweet depending on the chosen argument return_type
    """

    # convert to lower case
    tokens = [word.lower() for word in tweet_as_list_of_words]

    # rid 's, 've, 'm, 're, 'll, and 't
    tokens = [word if not word.endswith("'s") else word[:-2] for word in tokens]
    tokens = [word if not word.endswith("’s") else word[:-2] for word in tokens]
    tokens = [word if not word.endswith("'ve") else word[:-3] for word in tokens]
    tokens = [word if not word.endswith("’ve") else word[:-3] for word in tokens]
    tokens = [word if not word.endswith("'m") else word[:-2] for word in tokens]
    tokens = [word if not word.endswith("’m") else word[:-2] for word in tokens]
    tokens = [word if not word.endswith("'re") else word[:-3] for word in tokens]
    tokens = [word if not word.endswith("’re") else word[:-3] for word in tokens]
    tokens = [word if not word.endswith("'ll") else word[:-3] for word in tokens]
    tokens = [word if not word.endswith("’ll") else word[:-3] for word in tokens]
    tokens = [word if not word.endswith("'t") else word[:-2] for word in tokens]
    tokens = [word if not word.endswith("’t") else word[:-2] for word in tokens]

    # drop a few names, characters, and punctuations
    drop_list = ["barack", "hussein", "obama", "joe", "biden", "michelle", "malia", "ann", "natasha", "donald", "john",
                 "trump", "melania", "marla", "maples", "ivana", "ivanka", "jr.", "barron", "tiffany", "eric", "mike",
                 "pence", "'s", "’s", ".", ",", ":", "!", "(", ")", "[", "]", "{", "}", "~", "|", "`", "^", "?", ";",
                 "%", "&", "$", "'", '"', "*", "+", "<", "=", ">", "...", "-", "—", '“', '”', "’", "–", "’ve", "'ve",
                 "/", "’re", "'re", "’m", "'m", "’ll", "'ll", "’t", "'t"]
    tokens = [word for word in tokens if not word in drop_list]

    # drop stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if not word in stop_words]

    # drop @...
    tokens = [word for word in tokens if not word.startswith("@")]

    # drop #...
    tokens = [word for word in tokens if not word.startswith("#")]

    # drop http...
    tokens = [word for word in tokens if not word.startswith("http")]

    # drop numbers
    tokens = [word for word in tokens if match("^\d", word) is None]
    tokens = [word for word in tokens if match("^\+", word) is None]

    # no stemmed nor lemmed
    norm = " ".join(tokens)

    # PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    stemmed = " ".join(stemmed)

    # WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(word) for word in tokens]
    lemmed = " ".join(lemmed)

    # return
    if return_type == "stemmed":
        return stemmed
    elif return_type == "lemmed":
        return lemmed
    else:
        return norm


def clean_tweets(df):
    """
    Process the entire tweet data frame

    Arguments
    ---------
    :param df:  tweet data frame to process

    Return
    ------
    :return:    processed tweet data frame
    """

    # drop na
    df.dropna(inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # clean each individual tweet
    tknzr = TweetTokenizer()
    df["tweet_norm"] = df.tweet.apply(lambda tw: clean_each_tweet(tknzr.tokenize(tw)))
    df["tweet_stemmed"] = df.tweet.apply(lambda tw: clean_each_tweet(tknzr.tokenize(tw), "stemmed"))
    df["tweet_lemmed"] = df.tweet.apply(lambda tw: clean_each_tweet(tknzr.tokenize(tw), "lemmed"))

    # reset index
    df.reset_index(inplace=True, drop=True)

    # return
    return df
