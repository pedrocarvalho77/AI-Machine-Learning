import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def remove_columns(speedDating):
    #condição para verificar se existe missing values em mais de 70% das entradas para cada coluna
     #como em nenhuma situaçao tal se verifica nao eliminamos colunas
    total_rows=speedDating.shape[0]
    print("\nNumber of rows =",total_rows)
    print("\nTotal Nulls per column:\n\n" ,speedDating.isnull().sum())
    print("\nCondition to verify if there more then 70% of nulls per column\n" ,speedDating.isnull().sum() >= 0.7*total_rows)

def remove_row(speedDating):
    # Eliminar linhas
    row_speedDating = speedDating.dropna()
    row_speedDating.info()
    #remove cerca de 18% das linhas o que tendo em conta que sao apenas 8378
    #creio que sao demasiadas para este problema, assims sendo nao serao removidas linhas

def calc_mean_median_mode(my_list,speedDating,metric):
    print()
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            if metric == "mean":
                print("Mean",i,"=",speedDating[i].mean())
            elif metric == "median":
                print("Median",i,"=",speedDating[i].median())
            elif metric == "mode":
                print("Mode",i,":")
                print(speedDating[i].mode())
    print()

def replace_missing_values(my_list,speedDating):
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            speedDating[i] = speedDating[i].fillna(speedDating[i].mean())

def balanced(speedDating):
    positive = len(speedDating[speedDating.match==1])
    negative = len(speedDating[speedDating.match==0])
    print("Class 1 =",positive,"Class 0 =",negative)
    total = positive+negative
    if (positive/total < 0.45 or positive/total > 0.55):
        print("Data is unbalanced\n")
    else:
        print("Data is balanced\n")

def clf_holdout(X,y,clf,size):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=size,random_state=42)
    clf = clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    get_confusion_matrix(0,"clf",clf,"holdout",y_test,y_predict)
    print_holdout_scores(y_test,y_predict)

def clf_cross_validation(X,y,clf,size):
    kf = KFold(n_splits=size,random_state=42, shuffle=True)
    id = 1
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    for train_index, test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        clf.fit(X_train,y_train)
        y_predict = clf.predict(X_test)
        tn, fp, fn, tp = get_confusion_matrix(id,"clf",clf,"cv",y_test,y_predict)
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp
        id += 1
    print_cv_scores(total_tn,total_fp,total_fn,total_tp)

def gnb_holdout(X,y,gnb,size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
    y_predict = gnb.fit(X_train, y_train).predict(X_test)
    get_confusion_matrix(0,"gnb",gnb,"holdout",y_test,y_predict)
    print_holdout_scores(y_test,y_predict)

def gnb_cv(X,y,gnb,size):
    kf = KFold(n_splits=size,random_state=42, shuffle=True)
    id = 1
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    for train_index, test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        gnb.fit(X_train,y_train)
        y_predict = gnb.predict(X_test)
        tn, fp, fn, tp = get_confusion_matrix(id,"gnb",gnb,"cv",y_test,y_predict)
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp
        id += 1
    print_cv_scores(total_tn,total_fp,total_fn,total_tp)

def get_confusion_matrix(id,estimator_type,estimator,splitter_type,y_test,y_predict):
    if id == 0:
        id = ""
    else:
        id = str(id)
    cm = confusion_matrix(y_test,y_predict)
    tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_)
    cm_disp.plot(colorbar=False)
    file_path="/home/pedro/Faculdade/3ano/2sem/IA/trabalho2/AI-Machine-Learning/pictures/"+estimator_type+"/"
    if splitter_type == "cv":
        plt.savefig(file_path+splitter_type+"_confusion_matrix_"+id+".svg")
    elif splitter_type == "holdout":
        plt.savefig(file_path+splitter_type+"_confusion_matrix_"+id+".svg")
    return tn, fp, fn, tp

def print_holdout_scores(y_test,y_predict):
    print('Accuracy Score: %.3f' % accuracy_score(y_test,y_predict))
    print('Precision Score: %.3f' % precision_score(y_test,y_predict))
    print('Recall Score: %.3f' % recall_score(y_test,y_predict))
    print('F1 Score: %.3f' % f1_score(y_test,y_predict))

def print_cv_scores(tn,fp,fn,tp):
    accuracy = ((tp + tn)/(tp + fn + tn + fp))
    precision = ((tp)/(tp+fp))
    recall = ((tp)/(tp+fn))
    f1 = (2*(precision*recall)/(precision+recall))
    print('Accuracy Score: %.3f' % accuracy)
    print('Precision Score: %.3f' % precision)
    print('Recall Score: %.3f' % recall)
    print('F1 Score: %.3f' % f1)

def export_tree(type,depth,clf):
    features_names= ['id','partner','age','age_o','goal','date','go_out','int_corr','length','met','like','prob']
    target_names = ['0','1']
    dot = tree.export_graphviz(clf,max_depth=depth,out_file=None,feature_names=features_names,class_names=target_names,filled=True,rounded=True)
    graph = pydotplus.graph_from_dot_data(dot)
    Image(graph.create_png())
    file_path="/home/pedro/Faculdade/3ano/2sem/IA/trabalho2/AI-Machine-Learning/pictures/clf/"
    graph_name = file_path+type+"_tree_depth_"+str(depth)+".png"
    graph.write_png(graph_name)

def main():
    speedDating = pd.read_csv("~/Faculdade/3ano/2sem/IA/trabalho2/AI-Machine-Learning/speedDating_trab.csv")
    
    remove_columns(speedDating)
    remove_row(speedDating)
    
    my_list = list(speedDating)

    calc_mean_median_mode(my_list,speedDating,"mean")
    replace_missing_values(my_list,speedDating)

    X = speedDating[["id","partner","age","age_o","goal","date","go_out","int_corr","length","met","like","prob"]]
    y = speedDating["match"]
    
    balanced(speedDating)
    
    print("CLF Holdout")
    clf = tree.DecisionTreeClassifier()
    clf_holdout(X,y,clf,0.3)
    export_tree("holdout",6,clf)

    print("\CLF Cross Validation")
    clf = tree.DecisionTreeClassifier()
    clf_cross_validation(X,y,clf,5)
    export_tree("cv",6,clf)

    print("\nGNB Holdout")
    gnb = GaussianNB()
    gnb_holdout(X,y,gnb,0.5)
    
    print("\nGNB Cross Validation")
    gnb = GaussianNB()
    gnb_cv(X,y,gnb,5)

if __name__ == "__main__":
    main()
