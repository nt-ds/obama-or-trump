# --- Import libraries ---
import GetOldTweets3 as got
import pandas as pd


# --- Function ---
def get_tweets(username,
               start_date="2009-01-20",
               end_date="2020-04-25",
               emoji="ignore",
               max_tweets=7066):
    """
    Retrieve tweets using GetOldTweets3 (https://github.com/Mottl/GetOldTweets3)

    Arguments
    ---------
    :param username:    whose tweets to retrieve

    Optional Arguments
    ------------------
    :param start_date:  date to start retrieving tweets
                        default to President Obama's Inauguration date

    :param end_date:    date to stop retrieving tweets
                        default to a random recent date

    :param emoji:       whether or not to include emojis and how
                        default to "ignore"
                        can also choose ["name", "unicode"]

    :param max_tweets:  maximum number of tweets to retrieve
                        default to the number of tweets President Obama posted between the set dates
    """

    # create an empty data frame
    df = pd.DataFrame(columns=["tweet", "author"])

    # set criteria for tweets to be retrieved
    tweet_criteria = got.manager.TweetCriteria().setUsername(username) \
        .setSince(start_date) \
        .setUntil(end_date) \
        .setEmoji(emoji) \
        .setMaxTweets(max_tweets)

    # get tweets
    tweets = got.manager.TweetManager.getTweets(tweet_criteria)

    # append tweets to the data frame
    for tweet in tweets:
        df = df.append({"tweet": tweet.text, "author": username}, ignore_index=True)

    # save the data frame
    df.to_csv(
        "C:\\Users\\15713\\Desktop\\DS Projects\\Obama or Trump\\obama-or-trump\\data\\" + username + "_tweets.csv")
