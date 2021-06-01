import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt

def calc_mean_median_mode(my_list,speedDating):
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            print("Mean",i,"=",speedDating[i].mean())
            print("Median",i,"=",speedDating[i].median())
            print("Mode",i,":")
            print(speedDating[i].mode())

def replace_missing_values(my_list,speedDating):
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            speedDating[i] = speedDating[i].fillna(speedDating[i].mean())

def holdout(X,y,clf,size):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=size,random_state=42)
    clf = clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    get_confusion_matrix(clf,False,y_test,y_predict)
    print('Accuracy Score: %.3f' % accuracy_score(y_test,y_predict))
    print('Precision Score: %.3f' % precision_score(y_test,y_predict))
    print('Recall Score: %.3f' % recall_score(y_test,y_predict))
    print('F1 Score: %.3f' % f1_score(y_test,y_predict))

def cross_validation(X,y,clf,size):
    scoring = ['accuracy','precision','recall','f1']
    kf = KFold(n_splits=size,random_state=42, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        clf.fit(X_train,y_train)
        y_predict = clf.predict(X_test)
        #print("y_predict.sum() =",y_predict.sum())
        #print(y_predict[:15])
        #print(y_test.head(15))
        #print("y_test.sum() =",y_test.sum())
        get_confusion_matrix(clf,True,y_test,y_predict)
        scores = cross_validate(clf,X,y,cv=size,scoring=scoring)
    print_all_scores(scoring,scores,True)

def get_confusion_matrix(clf,cross_val,y_test,y_predict):
    cm = confusion_matrix(y_test,y_predict)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    cm_disp.plot(colorbar=False)
    if cross_val == True:
        plt.savefig('confusion_matrix_cv.svg')
    else:
        plt.savefig('confusion_matrix_holdout.svg')


def print_all_scores(scoring,scores,mean):
    for i in scoring:
        if mean == True:
            print(i,"score: %.3f" % scores['test_'+i].mean())
        else:
            print(i,scores['test_'+i].round(3))

def export_tree(depth,clf):
    features_names= ['id','partner','age','age_o','goal','date','go_out','int_corr','length','met','like','prob']
    target_names = ['0','1']
    dot = tree.export_graphviz(clf,max_depth=depth,out_file=None,feature_names=features_names,class_names=target_names,filled=True,rounded=True)
    graph = pydotplus.graph_from_dot_data(dot)
    Image(graph.create_png())
    graph_name = "tree_depth_"+str(depth)+".png"
    graph.write_png(graph_name)

def remove_columns(speedDating):
    #condição para verificar se existe missing values em mais de 70% das entradas para cada coluna
    total_rows=speedDating.shape[0]
    print("Number of rows =",total_rows)
    print("\nTotal Nulls per column\n" ,speedDating.isnull().sum())
    print("\nCondition to verify if there more then 70% of nulls per column\n" ,speedDating.isnull().sum() >0.7*total_rows)

    #como em nenhuma situaçao tal se verifica nao eliminamos colunas

def remove_row(speedDating):
    # Eliminar linhas
    row_speedDating = speedDating.dropna()
    row_speedDating.info()

    #remove cerca de 18% das linhas o que tendo em conta que sao apenas 8378
    #creio que sao demasiadas para este problema, assims sendo nao serao removidas linhas

def main():
    speedDating = pd.read_csv("~/Faculdade/3ano/2sem/IA/trabalho2/AI-Machine-Learning/speedDating_trab.csv")
    remove_columns(speedDating)
    remove_row(speedDating)
    
    my_list = list(speedDating)

    calc_mean_median_mode(my_list,speedDating)
    replace_missing_values(my_list,speedDating)

    X = speedDating[["id","partner","age","age_o","goal","date","go_out","int_corr","length","met","like","prob"]]
    y = speedDating["match"]

    clf = tree.DecisionTreeClassifier()

    x = input("Holdout or Cross Validation (h or cv)? ")
    if x == "h":
        holdout(X,y,clf,0.3)
    elif x=="cv":
        cross_validation(X,y,clf,5)

    export_tree(6,clf)

if __name__ == "__main__":
    main()
