from sklearn import linear_model,neighbors,svm,preprocessing,tree,ensemble,tree,metrics
import pandas
import numpy
import xgboost as xgb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

class trainning_netflix:
    def __init__(self,db,pct,col_pred,nb_couches,nb_var_select,nb_var_couche,cv,nb_split):
        self.db = db
        self.pct = pct
        self.x_train = pandas.DataFrame()
        self.x_test = pandas.DataFrame()
        self.y_train = pandas.DataFrame()
        self.y_test = pandas.DataFrame()
        self.col_pred = col_pred
        self.nb_couches = nb_couches
        self.nb_var_select = nb_var_select
        self.nb_var_couche = nb_var_couche
        self.cv = cv
        self.mes = pandas.DataFrame()
        self.mes['11'] = [0] * 10
        self.mes['10'] = 0
        self.mes['01'] = 0
        self.mes['00'] = 0
        self.mes['auc'] = 0
        self.nb_split = nb_split

    def split_train_test(self):
        rand = numpy.random.choice(range(len(self.db)),size = int((1 - self.pct) * len(self.db)),replace = False)
        self.x_train = self.db.iloc[rand]
        self.y_train = self.x_train[self.col_pred]
        self.x_train = self.x_train.drop(self.col_pred,axis = 1)
        self.x_test = self.db.drop(rand,axis = 0)
        self.y_test = numpy.array(self.x_test[self.col_pred])
        self.x_test = self.x_test.drop(self.col_pred,axis = 1)

    def th_evolution(self):
        for i in range(self.nb_couches):
            n_x_train = numpy.array([])
            n_x_test = numpy.array([])
            col_x = list(self.x_train.columns)
            for j in range(self.nb_var_couche[i]):
                col_rand = numpy.random.choice(col_x,size = min(len(col_x),self.nb_var_select),replace = False)
                for c in col_rand:
                    col_x.remove(c)
                model = linear_model.LinearRegression()
                # model = linear_model.LogisticRegression(penalty='none',solver='newton-cg')
                x_an_train = self.x_train[col_rand]
                x_an_test = self.x_test[col_rand]
                model.fit(x_an_train,self.y_train)
                pred_train = model.predict(x_an_train)[:,0]
                pred_test = model.predict(x_an_test)[:,0]
                if n_x_train.shape[0] == 0:
                    n_x_train = pred_train
                    n_x_test = pred_test
                else:
                    n_x_train = numpy.c_[n_x_train,pred_train]
                    n_x_test = numpy.c_[n_x_test,pred_test]
            self.x_test = pandas.DataFrame(n_x_test)
            self.x_train = pandas.DataFrame(n_x_train)
        model = linear_model.LogisticRegression(penalty='none',solver='newton-cg')
        model.fit(self.x_train,self.y_train)
        return model.predict_proba(self.x_test)[:,1]

    def mesure(self):
        for i in range(self.cv):
            self.split_train_test()
            k_y = 1
            y_pred = None
            for col_y in self.y_train:
                # cor_xy = []
                # for c in self.x_train:
                #     cor_xy.append(abs(self.y_train[col_y].corr(self.x_train[c])))
                # cor_xy = numpy.array(cor_xy)
                # sel = cor_xy.astype(str) == 'nan'
                # cor_xy[sel] = 0
                # cor_xy = pandas.DataFrame(cor_xy)
                # cor_xy.columns = ['corr']
                # cor_xy = cor_xy.sort_values(['corr'],ascending = False)
                k = 3
                col_x = self.x_train.columns
                
                # model = Sequential()
                # model.add(Dense(128, activation='relu'))
                # model.add(Dense(128, activation='softplus'))
                # model.add(Dense(128, activation='tanh'))
                # # model.add(Dropout(0.5))
                # model.add(Dense(1, activation='sigmoid'))
                # model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
                # model.fit(numpy.array(self.x_train), numpy.array(self.y_train[col_y]),batch_size=32, nb_epoch=10, verbose=0)
                # try:
                #     y_pred = numpy.c_[y_pred,model.predict_proba(self.x_test)[:,0]]
                # except:
                #     y_pred = model.predict_proba(self.x_test)[:,0]

                while k <= 3:
                #     x_an_train = self.x_train[col_x[cor_xy.index[0:int(k / self.nb_split * len(col_x))]]]
                #     x_an_test = self.x_test[col_x[cor_xy.index[0:int(k / self.nb_split * len(col_x))]]]
                    
                #     model = linear_model.LogisticRegression(penalty='none',solver='newton-cg')
                #     # model = tree.DecisionTreeClassifier()
                #     # model = ensemble.RandomForestClassifier()
                #     # model = svm.SVC(probability = True)
                #     # model = xgb.XGBClassifier(objective="binary:logistic")
                #     model.fit(x_an_train,self.y_train[col_y])
                #     try:
                #         y_pred = numpy.c_[y_pred,model.predict_proba(x_an_test)[:,1]]
                #     except:
                #         y_pred = model.predict_proba(x_an_test)[:,1]
                #     # fpr, tpr, thresholds = metrics.roc_curve(self.y_test, y_pred, pos_label=1)
                #     # auc_res = metrics.auc(fpr, tpr)
                #     # sel_11 = (y_pred >= 0.5) *  (self.y_test == 1)
                #     # sel_10 = (y_pred >= 0.5) *  (self.y_test == 0)
                #     # sel_01 = (y_pred < 0.5) * (self.y_test == 1)
                #     # sel_00 = (y_pred < 0.5) * (self.y_test == 0)
                #     # self.mes['11'].iloc[k-1] += sum(sel_11)
                #     # self.mes['10'].iloc[k-1] += sum(sel_10)
                #     # self.mes['01'].iloc[k-1] += sum(sel_01)
                #     # self.mes['00'].iloc[k-1] += sum(sel_00)
                #     # self.mes['auc'].iloc[k-1] += auc_res
                    print(str(k_y) + ' - ' + str(k) + ' - ' + str(i))
                    k += 1
                k_y += 1
            y_dec = numpy.apply_along_axis(numpy.argmax,1,y_pred)
            kyd = 0
            for d in y_dec:
                self.mes['11'].iloc[0] += self.y_test[kyd,d]
                self.mes['10'].iloc[0] += 1 - self.y_test[kyd,d]
                kyd += 1
        self.mes.to_csv('C:\\netflix\\model_outecomes.csv',sep = ';',index = False)
        # self.mes.to_csv('C:\\netflix\\' + self.col_pred + '.csv',sep = ';',index = False)

db = pandas.read_csv('C:\\netflix\\nlp.csv',engine = 'python',sep = ';')
pred = pandas.DataFrame()
col_pred = []
for c in db:
    if c.find('listed_in_') != -1:
        # pred[c] = numpy.array(db[c])
        # db.drop([c],axis = 1)
        col_pred.append(c)
m_db = db.shape[1]
nb_var_select = 10
nb_var_couche = [int((m_db - 0.01) / nb_var_select) + 1]
k = 2
while nb_var_couche[-1] >= nb_var_select:
    nb_var_couche.append(int((m_db - 0.01) / (numpy.power(nb_var_select, float(k)))) + 1)
    k += 1
nb_couches = len(nb_var_couche)
# for c in pred:
    # db[c] = numpy.array(pred[c])
tn = trainning_netflix(db = db,
                        pct = 0.1,
                        col_pred = col_pred,
                        nb_couches = nb_couches,
                        nb_var_select = nb_var_select,
                        nb_var_couche = nb_var_couche,
                        cv = 1,
                        nb_split = 10)
tn.mesure()
    # db.drop([c],axis = 1)
