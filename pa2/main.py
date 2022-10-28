from ast import Mult
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

review_files = load_files('./data/documents', encoding='latin-1')
x_train, x_test, y_train, y_test = train_test_split(review_files.data, review_files.target, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words='english')
x_trainvec = vectorizer.fit_transform(x_train)
x_testvec = vectorizer.transform(x_test)
# print(x_trainvec.toarray())
# print(vectorizer.get_feature_names_out())

mnb = MultinomialNB()
mnb.fit(x_trainvec, y_train)
y_preds = mnb.predict(x_testvec)

print('Length predictions', len(y_preds))
print('Length y_test', len(y_test))

correct_pred = 0
for i in range(len(y_test)):
    if y_preds[i] == y_test[i]:
        correct_pred += 1

print(f"Correct predictions {correct_pred} out of {len(y_test)}")

# bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)