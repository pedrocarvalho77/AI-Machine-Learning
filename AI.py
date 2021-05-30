import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from IPython.display import Image
import pydotplus

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
    """print("X_train\n",X_train)
    print("X_train\n",X_train.shape)
    print("y_train\n",y_train)
    print("X_test\n",X_test)
    print("X_test\n",X_test.shape)
    print("y_test\n",y_test)"""
    clf = clf.fit(X_train,y_train)

def cross_validation(clf,size):
    scoring = ['accuracy', 'precision','recall']
    scores = cross_validate(clf,X,y,cv=size,scoring=scoring)
    confusion_matrix()

    print(scores.keys())
    print_all_scores(scoring,scores,False)
 
def print_all_scores(scoring,scores,mean):
    for i in scoring:
        if mean == True:
            print(i,"=",scores['test_'+i].mean())
        else:
            print(i,scores['test_'+i])
  

def export_tree(depth):
    features_names= ['id','partner','age','age_o','goal','date','go_out','int_corr','length','met','like','prob']
    target_names = ['0','1']
    
    dot = tree.export_graphviz(clf,max_depth=depth,out_file=None,feature_names=features_names,class_names=target_names)
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
mod_speedDating = speedDating.dropna()
mod_speedDating.info()

#remove cerca de 18% das linhas o que tendo em conta que sao apenas 8378
#creio que sao demasiadas para este problema, assims sendo nao serao removidas linhas
my_list = list(speedDating)

calc_mean_median_mode()
replace_missing_values()

X = speedDating[["id","partner","age","age_o","goal","date","go_out","int_corr","length","met","like","prob"]]
y = speedDating["match"]

clf = tree.DecisionTreeClassifier()

holdout(clf,0.3)
cross_validation(clf,5)

"""
print(clf.get_depth())

print(clf.predict([[5,11,15,13,3,3,2,0.7,3,0,8,1]]))
print(clf.predict([[5,11,15,13,3,3,2,0.7,3,0,3,1]]))

print(clf.predict_proba([[5,11,15,13,3,3,2,0.7,3,0,8,1]]))
print(clf.predict_proba([[5,11,15,13,3,3,2,0.7,3,0,3,1]]))
"""

export_tree(4)

#funcoes para avaliar desempenho
#accuracy_score()
#error rate
#confusion_matrix()
#precision_recall_fscore_support()
#median_absolute_error()
#mean_squared_error()