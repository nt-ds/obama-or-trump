# obama-or-trump

### Directory structure
--- main folders and files only ---

		obama-or-trump
			data (raw and processed)
				BarackObama_tweets.csv		(retrieved)
				realDonaldTrump_tweets.csv	(retrieved)
				processed_tweets.csv		(processed)
			
			models (saved models and vectorizers)
				norm_vectorizer.pickle
				norm_naive_bayes_model.pickle
				norm_logistic_model.pickle
				norm_svm_model.pickle
				stemmed_vectorizer.pickle
				stemmed_naive_bayes_model.pickle
				stemmed_logistic_model.pickle
				stemmed_svm_model.pickle
				lemmed_vectorizer.pickle
				lemmed_naive_bayes_model.pickle
				lemmed_logistic_model.pickle
				lemmed_svm_model.pickle
						
			notebooks (executable code)
				create_data.py
				process_data.ipynb
				build_model.ipynb
				evaluate_model.ipynb
			
			src (source code)
				get_tweets.py
				clean_tweets.py
				model.py