import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt

def calc_mean_median_mode():
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            print("Mean",i,"=",speedDating[i].mean())
            print("Median",i,"=",speedDating[i].median())
            print("Mode",i,":")
            print(speedDating[i].mode())

def replace_missing_values():
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            speedDating[i] = speedDating[i].fillna(speedDating[i].mean())

def holdout(clf,size):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=size,random_state=42)
    clf = clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    get_confusion_matrix(False,y_test,y_predict)

def cross_validation(clf,size):
    kf = KFold(n_splits=size,random_state=42, shuffle=True)
    for train, test in kf.split(X):
        print("%s %s" % (train, test))
    clf = clf
    scoring = ['accuracy','precision','recall','f1']
    y_predict = cross_val_predict(clf,X,y,cv=size)
    #get_confusion_matrix(True,y_test,y_predict)
    scores = cross_validate(clf,X,y,cv=size,scoring=scoring)
    print_all_scores(scoring,scores,True)

def get_confusion_matrix(cross_val,y_test,y_predict):
    cm = confusion_matrix(y_test,y_predict)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    cm_disp.plot(colorbar=False)
    if cross_val == True:
        plt.savefig('confusion_matrix_cv.svg')
    else:
        plt.savefig('confusion_matrix_no_cv.svg')


def print_all_scores(scoring,scores,mean):
    for i in scoring:
        if mean == True:
            print(i,"=",scores['test_'+i].mean())
        else:
            print(i,scores['test_'+i])
  

def export_tree(depth):
    features_names= ['id','partner','age','age_o','goal','date','go_out','int_corr','length','met','like','prob']
    target_names = ['0','1']
    
    dot = tree.export_graphviz(clf,max_depth=depth,out_file=None,feature_names=features_names,class_names=target_names,filled=True,rounded=True)
    graph = pydotplus.graph_from_dot_data(dot)
    Image(graph.create_png())
    graph_name = "tree_depth_"+str(depth)+".png"
    graph.write_png(graph_name)

speedDating = pd.read_csv("~/Faculdade/3ano/2sem/IA/trabalho2/AI-Machine-Learning/speedDating_trab.csv")
#print(speedDating)

#condição para verificar se existe missing values em mais de 70% das entradas para cada coluna
total_rows=speedDating.shape[0]
print("Number of rows =",total_rows)
print("\nTotal Nulls per column\n" ,speedDating.isnull().sum())
print("\nCondition to verify if there more then 70% of nulls per column\n" ,speedDating.isnull().sum() >0.7*total_rows)

#como em nenhuma situaçao tal se verifica nao eliminamos colunas

# Eliminar linhas
row_speedDating = speedDating.dropna()
row_speedDating.info()

#remove cerca de 18% das linhas o que tendo em conta que sao apenas 8378
#creio que sao demasiadas para este problema, assims sendo nao serao removidas linhas
my_list = list(speedDating)

calc_mean_median_mode()
replace_missing_values()

X = speedDating[["id","partner","age","age_o","goal","date","go_out","int_corr","length","met","like","prob"]]
y = speedDating["match"]

clf = tree.DecisionTreeClassifier()

#holdout(clf,0.3)
cross_validation(clf,5)

#export_tree(6)