
***Context:***

The goal of this small project was to evaluate the performance of different machine learning models when applied to the problem of banknote authentication and the data set at hand. The data, available at the UCI Machine Learning Repository, is organized in a way that each row represents a banknote and each has 5 columns: 4 banknote features and a label classifying each banknote as real (value 0) or fake (value 1). Since this is a classification problem, the alghoritms were chosen accordingly and so the comparison ended up being between Logistic Regression, K-Nearest Neighbours and Naive Bayes.

After analysing the results given in the report folder and looking at the means obtained for model comparisions, it can be said with 95% degree of confidence that the K-Nearest Neighbours classifier has a significant performance impovement for this problem domain when compared to the other two. This conclusion is based on McNemar's test exceeding its reference value of 3.84 and the error estimate of K-NN being consistently lower than that of the other two. It's also possible to say that Logistic Regression is a better choice than Naive Bayes, for the same reasons.

***Running with Python:***

```
git clone https://github.com/danielmurteira/ml-banknotes.git
python banknote_authentication.py
```

***Running with Docker:***

```
git clone https://github.com/danielmurteira/ml-banknotes.git
docker build -t dmurteira/ml-banknotes .
docker run --name ml-banknotes -v %cd%/report:/home/report -d dmurteira/ml-banknotes
```

***Source:***

https://archive.ics.uci.edu/ml/datasets/banknote+authentication
