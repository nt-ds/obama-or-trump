# --- Import libraries ---
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle

sns.set_style("whitegrid")


# --- Functions ---
def get_train_test(df):
    """
    Split data frame into training and test sets

    Arguments
    ---------
    :param df:  tweet data frame

    Return
    ------
    :return:    training and test sets
    """

    # clean the data frame
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # train test split President Obama's part
    obama = df[df.author == "BarackObama"]
    obama.reset_index(inplace=True, drop=True)
    obama_train, obama_test = train_test_split(obama, test_size=0.2, random_state=8456)
    obama_train.reset_index(inplace=True, drop=True)
    obama_test.reset_index(inplace=True, drop=True)

    # train test split President Trump's part
    trump = df[df.author == "realDonaldTrump"]
    trump.reset_index(inplace=True, drop=True)
    trump_train, trump_test = train_test_split(trump, test_size=0.2, random_state=4632)
    trump_train.reset_index(inplace=True, drop=True)
    trump_test.reset_index(inplace=True, drop=True)

    # create training set
    training = pd.concat([obama_train, trump_train], ignore_index=True)
    training = shuffle(training, random_state=82734)
    training.reset_index(inplace=True, drop=True)

    # create test set
    test = pd.concat([obama_test, trump_test], ignore_index=True)
    test = shuffle(test, random_state=2374)
    test.reset_index(inplace=True, drop=True)

    # return
    return training, test


def vectorize_tweets(tweets, tweet_type):
    """
    Vectorize tweets

    Arguments
    ---------
    :param tweets:          tweets to vectorize

    :param tweet_type:      name of processed tweet used
                            can choose ["norm", "stemmed", "lemmed"]

    Return
    ------
    :return:                vectorized tweets
    """

    # create and fit vectorizer
    if os.path.exists(
            "C:\\Users\\15713\\Desktop\\DS Projects\\Obama or Trump\\obama-or-trump\\models\\" + tweet_type + "_vectorizer.pickle"):
        save_vectorizer = open(
            "C:\\Users\\15713\\Desktop\\DS Projects\\Obama or Trump\\obama-or-trump\\models\\" + tweet_type + "_vectorizer.pickle",
            "rb")
        vectorizer = pickle.load(save_vectorizer)
        save_vectorizer.close()
    else:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(tweets)

        # save vectorizer
        save_vectorizer = open(
            "C:\\Users\\15713\\Desktop\\DS Projects\\Obama or Trump\\obama-or-trump\\models\\" + tweet_type + "_vectorizer.pickle",
            "wb")
        pickle.dump(vectorizer, save_vectorizer)
        save_vectorizer.close()

    # vectorize tweets
    vectorized_tweets = vectorizer.transform(tweets)

    # return
    return vectorized_tweets


def build_model(tweet_train, author_train, tweet_type, model_type, k_fold=10):
    """
    Build the best model to classify tweets (binary: written by either President Obama or President Trump)

    Arguments
    ---------
    :param tweet_train:     tweets to train on (X in training data)

    :param author_train:    authors to train on (y in training data)

    :param tweet_type:      name of processed tweet used
                            can choose ["norm", "stemmed", "lemmed"]

    :param model_type:      which model to build
                            can choose ["logistic" (Logistic Regression),
                                        "naive_bayes" (Gaussian Naive Bayes),
                                        "svm" (Support Vector Machine)]

    Optional Arguments
    ------------------
    :param k_fold:          number of folds (k-fold cross-validation)
                            default to 10

    Return
    ------
    :return:                the best model depending on the chosen argument model
    """

    # get model
    models = {"logistic": LogisticRegression(random_state=2374, n_jobs=-1),
              "naive_bayes": GaussianNB(),
              "svm": SVC(random_state=2374)}
    model = models[model_type]

    # vectorize tweets
    vectorized_tweet_train = vectorize_tweets(tweet_train, tweet_type)

    # set parameters for grid search
    param_grid = {}
    if model_type == "logistic":
        param_grid["C"] = [0.01, 0.1, 1.0, 10.0]
        param_grid["solver"] = ["saga", "sag", "lbfgs"]
        param_grid["warm_start"] = [True, False]
        save_name = "logistic"
    elif model_type == "svm":
        param_grid["C"] = [0.01, 0.1, 1.0, 10.0]
        param_grid["kernel"] = ["linear", "poly", "rbf", "sigmoid"]
        param_grid["degree"] = [3, 4, 5]
        param_grid["gamma"] = ["scale", "auto"]
        save_name = "svm"
    else:
        save_name = "naive_bayes"
        vectorized_tweet_train = vectorized_tweet_train.toarray()

    # create and fit grid search
    kfold = KFold(n_splits=k_fold, random_state=5765)
    clf = GridSearchCV(model, param_grid, cv=kfold, scoring="roc_auc", n_jobs=-1)
    clf.fit(vectorized_tweet_train, author_train)

    # evaluate grid search
    print("Best estimator:")
    print(clf.best_estimator_)
    print()

    print("Best parameters:")
    print(clf.best_params_)
    print()

    print("Best score:")
    print(clf.best_score_)
    print()

    print("Grid scores on training set:")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    params_sets = clf.cv_results_["params"]
    for mean, std, params in zip(means, stds, params_sets):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    # save classifier
    save_classifier = open(
        "C:\\Users\\15713\\Desktop\\DS Projects\\Obama or Trump\\obama-or-trump\\models\\" + tweet_type + "_" + save_name + "_model.pickle",
        "wb")
    pickle.dump(clf, save_classifier)
    save_classifier.close()

    # return
    return clf


def evaluate_model(model, tweet_test, author_test, test_df, tweet_type, is_naive_bayes=True):
    """
    Evaluate model

    Arguments
    ---------
    :param model:               model to evaluate

    :param tweet_test:          tweets to test (X in test data)

    :param author_test:         authors to test (y in test data)

    :param test_df:             test tweet data frame

    :param tweet_type:          name of processed tweet used
                                can choose ["norm", "stemmed", "lemmed"]

    Optional Arguments
    ------------------
    :param is_naive_bayes:      is the model to evaluate Gaussian Naive Bayes
                                default to True
                                can choose [True, False]
    """

    # vectorize tweets
    vectorized_tweet_test = vectorize_tweets(tweet_test, tweet_type)

    # format test tweets
    if is_naive_bayes:
        vectorized_tweet_test = vectorized_tweet_test.toarray()

    # evaluate model
    predicted = model.predict(vectorized_tweet_test)

    print("Classification Report")
    print("---------------------")
    report = classification_report(author_test, predicted)
    print(report)
    print()

    print("Plot Confusion Matrix")
    print("---------------------")
    titles_options = [("Non-Normalized Confusion Matrix", None),
                      ("Normalized Confusion Matrix", "true")]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, vectorized_tweet_test, author_test,
                                     cmap=plt.cm.Blues, normalize=normalize)
        plt.grid(False)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
        print()
    plt.show()
    print()

    print("Wrongly Classified")
    print("------------------")
    indices = [i for i in range(len(author_test)) if author_test[i] != predicted[i]]
    wrong_predictions = test_df.iloc[indices, :]
    wrong_predictions.reset_index(inplace=True, drop=True)
    display(wrong_predictions)                                      # display a data frame in a jupyter notebook
