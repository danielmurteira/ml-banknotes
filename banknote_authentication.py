import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors._kde import KernelDensity
from sklearn.metrics import accuracy_score

def load_data(file_name):
    matrix = np.loadtxt(file_name, delimiter=',')    
    return matrix

def calc_fold_logreg(feats,X,Y,train_ix,valid_ix,C=1e12):
    '''return classification error for train and validation sets'''
    reg = LogisticRegression(penalty='l2',C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2 
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

def calc_fold_knn(feats,X,Y,train_ix,valid_ix,k):
    '''return classification error for train and validation sets'''
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X[train_ix,:feats],Y[train_ix])
    error_r = 1 - neigh.score(X[train_ix,:feats],Y[train_ix])
    error_v = 1 - neigh.score(X[valid_ix,:feats],Y[valid_ix])
    return error_r,error_v

def nb_priors(X,Y):
    X0 = X[Y==0,:]
    X1 = X[Y==1,:]
    prob0 = np.log(X0.shape[0]/X.shape[0])
    prob1 = np.log(X1.shape[0]/X.shape[0])
    return X0,X1,prob0,prob1

def nb_kdes(feats,X0,X1,h):
    '''return matrix with kdes for every feature-class pair'''
    kdes = []
    for feature in range(feats):
        kde0 = KernelDensity(bandwidth=h,kernel='gaussian')
        kde0.fit(X0[:,[feature]])
        kde1 = KernelDensity(bandwidth=h,kernel='gaussian')
        kde1.fit(X1[:,[feature]])
        kdes.append((kde0,kde1))
    return kdes

def nb_predict(X,prob0,prob1,kdes):
    '''returns vector with class predictions for every point in the data set'''
    pred = np.zeros(X.shape[0])
    p0 = prob0
    p1 = prob1
    
    for feat in range(X.shape[1]):
        #score_samples returns vector with log of conditional probs
        #note: value + vector applies value sum to all values of the vector
        p0 += kdes[feat][0].score_samples(X[:,[feat]])
        p1 += kdes[feat][1].score_samples(X[:,[feat]])
        
    for row in range(X.shape[0]):
        if (p0[row] < p1[row]):
            pred[row] = 1
    
    return pred

def calc_fold_nb(feats,X,Y,train_ix,valid_ix,h):
    '''return classification error for train and validation sets'''
    X0,X1,prob0,prob1 = nb_priors(X[train_ix,:feats],Y[train_ix])
    kdes = nb_kdes(feats,X0,X1,h)
    score_r = accuracy_score(Y[train_ix],nb_predict(X[train_ix,:feats],prob0,prob1,kdes))
    score_v = accuracy_score(Y[valid_ix],nb_predict(X[valid_ix,:feats],prob0,prob1,kdes))
    error_r = 1 - score_r
    error_v = 1 - score_v
    return error_r,error_v

def create_errors_plot(best_x, scale_x, train_err, val_err, identifier):
    '''plot the training and validation errors'''
    title = 'picking best ' + identifier + '\n[best = ' + str(best_x) + ']'
    plt.title(title, fontsize=15)
    plt.xlabel(identifier, fontsize=12)
    plt.ylabel('error', fontsize=12)
    plt.plot(scale_x, train_err, color="red", linewidth=1.0)
    plt.plot(scale_x, val_err, color="black", linewidth=1.0)
    
    red_patch = mpatches.Patch(color='red', label='Training Error')
    black_patch = mpatches.Patch(color='black', label='Validation Error')
    plt.legend(handles=[red_patch, black_patch])
    
    filename = 'report/best_' + identifier + '.png'
    plt.savefig(filename, dpi=300)
    #plt.show()
    plt.close()
    
def mcnemars_e(pred_a,pred_b,classes):
    '''returns number of correct predictions in a that are wrong in b'''
    total = 0;
    for n in range(pred_a.shape[0]):
        if pred_a[n] == classes[n] and pred_b[n] != classes[n]:
            total = total + 1
    return total

#load data
mat = load_data("data/dataset.csv")

#shuffle data
data = shuffle(mat)

#standardize features
Ys = data[:,-1]
Xs = data[:,:-1]
means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)
Xs = (Xs-means)/stdevs

#split data
X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)

#cross validation for regularization: picking the best c
best_c = c = 1
best_err = 1000000
train_err = []
val_err = []
Cs = []

folds = 5
kf = StratifiedKFold(n_splits=folds)
for n in range(1,21):
    tr_err = va_err = 0
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        r,v = calc_fold_logreg(Xs.shape[1],X_r,Y_r,tr_ix,va_ix,c)
        tr_err += r
        va_err += v
    if (va_err/folds<best_err):
        best_err = va_err/folds
        best_c = np.log2(c)
    train_err.append(tr_err/folds)
    val_err.append(va_err/folds)
    Cs.append(np.log2(c))
    c = c*2

create_errors_plot(best_c, Cs, train_err, val_err, 'c')

#cross validation for kNN: picking the best k
best_k = 1
best_err = 1000000
train_err = []
val_err = []
Ks = []

folds = 5
kf = StratifiedKFold(n_splits=folds)
for k in range(1,40,2):
    tr_err = va_err = 0
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        r,v = calc_fold_knn(Xs.shape[1],X_r,Y_r,tr_ix,va_ix,k)
        tr_err += r
        va_err += v
    if (va_err/folds<best_err):
        best_err = va_err/folds
        best_k = k
    train_err.append(tr_err/folds)
    val_err.append(va_err/folds)
    Ks.append(k)

create_errors_plot(best_k, Ks, train_err, val_err, 'k')

#cross validation for naive bayes
best_h = 1
best_err = 1000000
train_err = []
val_err = []
Hs = []

folds = 5
kf = StratifiedKFold(n_splits=folds)
for h in range(1,100,2):
    h = h/100
    tr_err = va_err = 0
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        r,v = calc_fold_nb(Xs.shape[1],X_r,Y_r,tr_ix,va_ix,h)
        tr_err += r
        va_err += v
    if (va_err/folds<best_err):
        best_err = va_err/folds
        best_h = h
    train_err.append(tr_err/folds)
    val_err.append(va_err/folds)
    Hs.append(h)

create_errors_plot(best_h, Hs, train_err, val_err, 'h')

#mcnemars test
#predict class values for test sets using the different classifiers
reg = LogisticRegression(penalty='l2',C=best_c, tol=1e-10)
reg.fit(X_r,Y_r)
logreg_pred = reg.predict(X_t)
logreg_error = 1 - reg.score(X_t,Y_t)

neigh = KNeighborsClassifier(n_neighbors=best_k)
neigh.fit(X_r,Y_r)
knn_pred = neigh.predict(X_t)
knn_error = 1 - neigh.score(X_t,Y_t)

X0,X1,prob0,prob1 = nb_priors(X_r,Y_r)
kdes = nb_kdes(X_r.shape[1],X0,X1,best_h)
nb_pred = nb_predict(X_t,prob0,prob1,kdes)
nb_error = 1 - accuracy_score(Y_t,nb_pred)

#logreg vs knn
e01 = mcnemars_e(logreg_pred,knn_pred,Y_t)
e10 = mcnemars_e(knn_pred,logreg_pred,Y_t)
res1 = (float(abs(e01-e10)-1))**2 / (e01+e10)

#logreg vs nb
e01 = mcnemars_e(logreg_pred,nb_pred,Y_t)
e10 = mcnemars_e(nb_pred,logreg_pred,Y_t)
res2 = (float(abs(e01-e10)-1))**2 / (e01+e10)

#knn vs nb
e01 = mcnemars_e(knn_pred,nb_pred,Y_t)
e10 = mcnemars_e(nb_pred,knn_pred,Y_t)
res3 = (float(abs(e01-e10)-1))**2 / (e01+e10)

#analysis = significant difference between the 2 classifiers if res >= 3.84
f = open("report/report.txt", "w")
f.write("Parameter | Mean\n")
f.write("------------ | -------------\n")
f.write("Best C | " + str(best_c) + "\n")
f.write("Best K | " + str(best_k) + "\n")
f.write("Best H | " + str(best_h) + "\n")
f.write("\n")
f.write("Error | Mean\n")
f.write("------------ | -------------\n")
f.write("Logistic Regression | " + str(round(logreg_error,5)) + "\n")
f.write("KNN | " + str(round(knn_error,5)) + "\n")
f.write("Naive Bayes | " + str(round(nb_error,5)) + "\n")
f.write("\n")
f.write("McNemar | Mean\n")
f.write("------------ | -------------\n")
f.write("LogReg vs KNN | " + str(round(res1,5)) + "\n")
f.write("LogReg vs NB | " + str(round(res2,5)) + "\n")
f.write("KNN vs NB | " + str(round(res3,5)) + "\n")
f.close()
