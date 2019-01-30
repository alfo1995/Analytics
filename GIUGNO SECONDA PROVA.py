
# coding: utf-8

# # Analytics seconda prova Giugno

# Il compito consiste nell’analizzare il dataset Telekom  ed ottenere un modello predittivo in grado di individuare la classe di appartenenza (STATUS, binaria). La legenda è nel file txt  allegato.
# 
# Le fasi dell’analisi consistono in:
# 
# 1) Preprocessing dei dati (trasformazioni, ricodifiche, selezione) imputare le variabili che hanno missing
# 
# 2) Individuazione del modello ottimale, scegliendo tra quelli conosciuti (Alberi, gradient boosting, foreste random, logistico, reti neurali, ….)
# 
# 3) Risultati di valutazione(a scelta tra hold-out o cross-validation): matrici di confusione e AUC nel training e nel test (o cross-validation), curva ROC.
# 
# 4) Provare a bilanciare la target e verificare la differenza nei risultati 
# 
# 5) Valutare sinteticamente (10 righe) i risultati ottenuti
# 
# **I risultati e i commenti devono essere messi in un file .doc di cui dovete effettuare l'upload**

# Importo le librerie di alto livello di cui avrò bisogno

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import cross_val_score,cross_val_predict,train_test_split
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,classification_report
from keras import Sequential
from keras.utils import np_utils
from keras.layers import Activation,BatchNormalization,Dropout,Dense
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from keras.optimizers import Adam
get_ipython().magic('matplotlib inline')


# importo il dataset

# In[14]:


dataset=pd.read_csv('./telekom.csv',delimiter=',')


# In[15]:


#guardo le prime righe del dataset e rimuovo la variabile ID inutile ai fini della classificazione.
dataset=dataset.drop('ID',axis=1)
dataset.head()


# **Vediamo se ci sono variabili con Na's**

# In[16]:


print(dataset.isnull().values.any())
print(dataset.columns[dataset.isnull().any()].tolist())


# In[17]:


len(dataset.columns)


# Leggendo la descrizione delle variabili emerge che:
# - non ci sono variabili qualitative sul quale applicare Label Encoder
# - Le variabili (features) categoriche sul quale applicare l'encoding sono:
#         + piano_tariff
#         + metodo_pagamento
#         + sesso
#         + zona_attivaz
#         + canale_attivaz
#         + vas1
#         + vas2
# - Le restanti variabili sono numeriche 
# - La variabile 'status' è la nostra **variabile risposta $y$**

# ***encoding***

# In[18]:


dataset=pd.get_dummies(dataset,columns=['piano_tariff','metodo_pagamento','sesso'
                                        ,'zona_attivaz','canale_attivaz','vas1','vas2'])


# In[19]:


len(dataset.columns)


# Ora divido il dataset in $X$ ed $y$

# In[21]:


X=dataset.drop('status',axis=1).values
y=dataset['status'].values


# In[24]:


#provo a selezionare con SelectKbest, le 20 variabili migliori tramite il calcolo del chi2
#X_new = SelectKBest(chi2, k=20).fit_transform(X, y)


# A questo punto utilizzo la tecnica dell'HOLD OUT per dividere il dataset in Train e Test.

# In[39]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3)


# Ho deciso di utilizzare tre modelli e confrontarli tra loro:
# - Random Forest classifier
# - Gradient Boosting classifier
#     - eventualmente faccio il tuning dei parametri con Grid Search
# - Neural network
# 

# In[40]:


from sklearn.preprocessing import StandardScaler
#scale the X coloumns
sc_X = StandardScaler()
#for the training set we need to fit it, then scale it
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# **RANDOM FOREST CLASSIFIER**

# In[42]:



RFC=RandomForestClassifier()
RFC.fit(X_train,y_train)
y_pred=RFC.predict(X_test)

#calcolo accuracy
accuracy=accuracy_score(y_pred,y_test)
print('Accuracy: '+str(accuracy))
print()
print('Confusion matrix:')
cm=confusion_matrix(y_pred,y_test)
print(cm)

y_pred_proba = RFC.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print()
print("auc: "+str(auc))
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve Random Forest')
plt.legend(loc=4)
plt.show()

RF_out=[accuracy,auc]


# **GRADIENT BOOSTING**

# In[43]:


GB=GradientBoostingClassifier()
GB.fit(X_train,y_train)
y_pred=GB.predict(X_test)

#calcolo accuracy
accuracy=accuracy_score(y_pred,y_test)
print('Accuracy: '+str(accuracy))
print()
print('Confusion matrix:')
cm=confusion_matrix(y_pred,y_test)
print(cm)

y_pred_proba = RFC.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print()
print("auc: "+str(auc))
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve Random Forest')
plt.legend(loc=4)
plt.show()

GB_out=[accuracy,auc]


# **RETE NEURALE**

# In[48]:


model = Sequential()
model.add(Dense(8,input_dim=X_train.shape[1],activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.3))
model.add(Dense(8,activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(8,activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, y_train,epochs=200,batch_size=64)



y_pred=model.predict(X_test)
y_pred1=[]
for i in range(0,len(y_pred)):
    y_pred1.append(np.round(y_pred[i][0]))
y_pred=np.array(y_pred1,dtype=np.int64)
accuracy=accuracy_score(y_pred,y_test)
print('accuracy: %s'%accuracy)
print()
print('confusion matrix: ')
cm=confusion_matrix(y_pred,y_test)
print(cm)

nn_out=[accuracy,'NA']


# In[49]:


compare=pd.DataFrame(list(zip(RF_out,GB_out,nn_out,)),
                     index=pd.Series(['accuracy','auc'], name='metrics'),
              columns=['Random Forest','Gradient boosting','Neural network'])


# In[50]:


compare


# **Ora provo a fare il balancing del train e vedere se migliora la prediction**

# In[51]:


from imblearn.over_sampling import SMOTE
from sklearn.externals.joblib.parallel import _backend

sm = SMOTE(random_state=0, ratio = 1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)


# **RANDOM FOREST**

# In[55]:


RFC=RandomForestClassifier()
RFC.fit(X_train,y_train)
y_pred=RFC.predict(X_test)

#calcolo accuracy
accuracy=accuracy_score(y_pred,y_test)
print('Accuracy: '+str(accuracy))
print()
print('Confusion matrix:')
cm=confusion_matrix(y_pred,y_test)
print(cm)

y_pred_proba = RFC.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print()
print("auc: "+str(auc))
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve Random Forest')
plt.legend(loc=4)
plt.show()

RF_out_balance=[accuracy,auc]


# **GRADIENT BOOSTING**

# In[56]:


GB=GradientBoostingClassifier()
GB.fit(X_train,y_train)
y_pred=GB.predict(X_test)

#calcolo accuracy
accuracy=accuracy_score(y_pred,y_test)
print('Accuracy: '+str(accuracy))
print()
print('Confusion matrix:')
cm=confusion_matrix(y_pred,y_test)
print(cm)

y_pred_proba = RFC.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print()
print("auc: "+str(auc))
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve Random Forest')
plt.legend(loc=4)
plt.show()

GB_out_balance=[accuracy,auc]


# **RETE NEURALE**

# In[57]:


model = Sequential()
model.add(Dense(8,input_dim=X_train.shape[1],activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.3))
model.add(Dense(8,activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(8,activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, y_train,epochs=200,batch_size=64)



y_pred=model.predict(X_test)
y_pred1=[]
for i in range(0,len(y_pred)):
    y_pred1.append(np.round(y_pred[i][0]))
y_pred=np.array(y_pred1,dtype=np.int64)
accuracy=accuracy_score(y_pred,y_test)
print('accuracy: %s'%accuracy)
print()
print('confusion matrix: ')
cm=confusion_matrix(y_pred,y_test)
print(cm)

nn_out_balance=[accuracy,'NA']


# In[58]:


compare=pd.DataFrame(list(zip(RF_out,GB_out,nn_out,RF_out_balance,GB_out_balance,nn_out_balance)),
                     index=pd.Series(['accuracy','auc'], name='metrics'),
              columns=['Random Forest','Gradient boosting','Neural network',
              'Random Forest balanced','Gradient boosting balanced','Neural network balanced'])


# In[59]:


compare


# **CROSS VALIDATION**

# In[67]:


from sklearn.cross_validation import cross_val_score
RFC = RandomForestClassifier()
scores = cross_val_score(RFC, X, y, cv=5)
# stampe
print('\n Stampa dei risultati ')
print('Tree k-fold CV-error: %f' % scores.mean())

# k-fold cross-validation --> stima Y
predicted = cross_val_predict(RFC, X, y, cv=5)

print("Confusion matrix:\n%s" % confusion_matrix(y, predicted))


# In[66]:


from sklearn.cross_validation import cross_val_score
RFC = GradientBoostingClassifier()
scores = cross_val_score(RFC, X, y, cv=5)
# stampe
print('\n Stampa dei risultati ')
print('Tree k-fold CV-error: %f' % scores.mean())

# k-fold cross-validation --> stima Y
predicted = cross_val_predict(RFC, X, y, cv=5)

print("Confusion matrix:\n%s" % confusion_matrix(y, predicted))


# **COMMENTI**
# ... brevi cenni sull'analisi fatta..
# 
# Gli algoritmi applicati sono essenzaialmente tre:
# + Random Forest
# + Gradien boosting
# + rete neurale
# 
# Una volta fatto il fitting dei dati mediante i tre algoritmi abbiamo riportato in tabella per ognuno di essi quelle che sono le metriche di confronto più comuni:
# - Accuracy
# - Roc curve e auc (area under the curve)
# 
# I tre algoritmi sono stati provati su:
# + dataset pulito, splittato in train e test (HOLD OUT)
# + dataset pulito, splittato in train e test (HOLD OUT) con train bilanciato
# + dataset pulito, non splittato in train e test, ma utilizzando la cross validation
# 
# Ho scelto di comparare la "semplicità" e la maggior interpretabilità dei "Decision trees" con la complessità delle reti neurali, spesso definite come "Black box" per il procedimento non limpido che si cela dietro la classificazione.
# 
# Sicuramente sia il *Random forest* che il *Gradient boosting* sono più performanti rispetto ai  classici decision trees in quanto uno step fondamentale di questi algoritmi è il **Bagging**:
# - cioè vengono presi piu samples di Bootstrap dal dataset (train) e vengono creati degli alberi e quindi fatto il fit per ognuno di essi e poi fatta la media delle prediction degli  "M1,...,Mb" modelli.
# 
# Bootstrap + aggregation $\rightarrow$ Bagging
# 
# Nonostante le elevate performance di entrambi i modelli, la rete neurale sembra comunque fare una predizione più accurata.



##nel caso di target con piu di due classi (serve solo per reti neurali)!!
y=np_utils.to_categorical(y)


## tuning dei modelli
from sklearn.model_selection import GridSearchCV
classifier=RandomForestClassifier()
#specify the parameters for wich we want to find the optimal values, give different options in dictionaries
parameters = [{'n_estimators': [200,400,1000], "max_depth":[1,3,5],'min_samples_leaf':[1,2], 'min_samples_split':[2,4]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print('best_accuracy: %s'%best_accuracy)
print()
best_parameters = grid_search.best_params_
print('best_parameters: ')
print(best_parameters)