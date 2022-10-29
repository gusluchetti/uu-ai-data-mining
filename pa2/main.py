from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Loading the data and making the train-test splits.

review_files = load_files('./cleaned_op_spam', encoding='latin-1')
x_train, x_test, y_train, y_test = train_test_split(review_files.data, review_files.target, test_size=0.2)
# print(review_files)

# Since we're working with text data, we need to vectorize the documents / turn each document
# into a point vector in a n-dimentions space. 
# Moreover, rather than simply using the word counts, we want to use the Term Frequency as feature value.

vectorizer = TfidfVectorizer(stop_words='english')
x_trainvec = vectorizer.fit_transform(x_train)
x_testvec = vectorizer.transform(x_test)
# print(x_trainvec.toarray())
# print(vectorizer.get_feature_names_out())

mnb = MultinomialNB()
mnb.fit(x_trainvec, y_train)
y_preds = mnb.predict(x_testvec)
# print('Length predictions', len(y_preds))
# print('Length y_test', len(y_test))

correct_pred = 0
for i in range(len(y_test)):
    if y_preds[i] == y_test[i]:
        correct_pred += 1

print(f"Correct predictions {correct_pred} out of {len(y_test)}")

x_train_vec = vectorizer.fit_transform(review_files.data)
y_train_vec = review_files.target

print("Mean accuracy of MNB:", cross_val_score(mnb, x_train_vec, y_train_vec, cv=5, scoring='accuracy').mean())

logreg = LogisticRegression()
print("Mean accuracy of LogReg:", cross_val_score(logreg, x_train_vec, y_train_vec, cv=5, scoring='accuracy').mean())

dtree = DecisionTreeClassifier()
print("Mean accuracy of LogReg:", cross_val_score(logreg, x_train_vec, y_train_vec, cv=5, scoring='accuracy').mean())

ranForest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
print("Mean accuracy of Random Forest:", cross_val_score(ranForest, x_train_vec, y_train_vec, cv=5, scoring='accuracy').mean())