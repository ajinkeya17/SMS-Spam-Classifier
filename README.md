An end to end sms spam classifier model is created and deployed using Streamlit

## Dataset Used : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Phase 1.) Model design
The steps performed on the dataset using the jupyter notebook.
### 1.) Data Cleaning

removing irrelevant columns

encoding categorical labels in target column

dealing with nulls

dealing with duplicates

### 2.) EDA

### 3.) Text Preprocessing

Lower case

Tokenization

Removing special characters

Removing stop words and punctuations

Stemming/Lemmatizetion

### 4.) Text Vectorization using Bag of Words and Tfidf

### 5.) Model building
We experiment with multiple models such as Naive Bayes, Ensemble methods, SVM ,etc
To determine best performing model the following optimizations were implemented and results for all models records:
##### Optimizations :

a.) changing the max_features paramater of Tfidf to limit the vector length which may prevent overfitting

b.) apply min-max scalar to input data

c.) include the custom "number of characters" column we had created in our input vector

d.) Creating an ensemble classifier using two highest precision models after our above 3 experiments

After all experimentations the best performance is by Multinomial Naive Bayes Classifier keeping max_features as 3000 in tfidf vectorizer. The accuracy is 98.35% and precision is 1.0 (we use precision as we have an imblanaced dataset)

### 6.) Pickling : The vectorizer and best performing model are pickled to be deployed using Streamlit


## Phase 2.) Streamlit based model deployment
The pickled vectorizer is used to vectorize any new inputs entered by the user and the trained pickle model is used to make predictions. Streamlit based website deployed and tested.


For future reference and similar implementation see : https://www.youtube.com/watch?v=YncZ0WwxyzU&feature=emb_title
