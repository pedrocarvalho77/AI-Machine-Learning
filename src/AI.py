from pathlib import Path

import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
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
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=size,random_state=42,stratify=y)
    clf = clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    get_confusion_matrix(0,"clf",clf,"holdout",y_test,y_predict)
    print_holdout_scores(y_test,y_predict)

def clf_cross_validation(X,y,clf,size):
    kf = StratifiedKFold(n_splits=size,random_state=42, shuffle=True)
    id = 1
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    for train_index, test_index in kf.split(X,y):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42, stratify=y)
    y_predict = gnb.fit(X_train, y_train).predict(X_test)
    get_confusion_matrix(0,"gnb",gnb,"holdout",y_test,y_predict)
    print_holdout_scores(y_test,y_predict)

def gnb_cv(X,y,gnb,size):
    kf = StratifiedKFold(n_splits=size,random_state=42, shuffle=True)
    id = 1
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    for train_index, test_index in kf.split(X,y):
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

def model(estimator):
    # imputation inside the pipeline: means are computed on the training data only
    return make_pipeline(SimpleImputer(strategy="mean"), estimator)

ROOT = Path(__file__).resolve().parent.parent

def picture_dir(name):
    d = ROOT / "pictures" / name
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_confusion_matrix(id,estimator_type,estimator,splitter_type,y_test,y_predict):
    if id == 0:
        id = ""
    else:
        id = str(id)
    cm = confusion_matrix(y_test,y_predict)
    tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_)
    cm_disp.plot(colorbar=False)
    file_path=str(picture_dir(estimator_type))+"/"
    for ext in ("svg","pdf"):
        plt.savefig(file_path+splitter_type+"_confusion_matrix_"+id+"."+ext)
    plt.close()
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
    dot = tree.export_graphviz(clf[-1],max_depth=depth,out_file=None,feature_names=features_names,class_names=target_names,filled=True,rounded=True)
    graph = pydotplus.graph_from_dot_data(dot)
    file_path=str(picture_dir("clf"))+"/"
    graph_name = file_path+type+"_tree_depth_"+str(depth)+".png"
    graph.write_png(graph_name)

def extra_experiments(speedDating,features):
    # 5-fold stratified CV, imputation inside the pipeline
    no_id = [f for f in features if f not in ("id","partner")]
    y = speedDating["match"]
    runs = [
        ("CART, depth 5", tree.DecisionTreeClassifier(max_depth=5,random_state=42), features),
        ("CART, depth 5, balanced", tree.DecisionTreeClassifier(max_depth=5,class_weight="balanced",random_state=42), features),
        ("CART, balanced", tree.DecisionTreeClassifier(class_weight="balanced",random_state=42), features),
        ("CART, no id/partner", tree.DecisionTreeClassifier(random_state=42), no_id),
        ("GNB, no id/partner", GaussianNB(), no_id),
    ]
    kf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    for name,est,cols in runs:
        p = cross_val_predict(model(est),speedDating[cols],y,cv=kf)
        print("%-26s acc %.3f  prec %.3f  rec %.3f  f1 %.3f" % (name,accuracy_score(y,p),precision_score(y,p),recall_score(y,p),f1_score(y,p)))

def grouped_experiments(speedDating,features):
    # CV grouped by id (only 22 distinct values, so a coarse proxy for "same participant"); the one null id is its own group
    no_id = [f for f in features if f not in ("id","partner")]
    y = speedDating["match"]
    groups = speedDating["id"].fillna(-1)
    kf = StratifiedGroupKFold(n_splits=5,random_state=42,shuffle=True)
    for name,est,cols in (("CART, grouped by id",tree.DecisionTreeClassifier(random_state=42),features),
                          ("CART, grouped, no id/partner",tree.DecisionTreeClassifier(random_state=42),no_id),
                          ("GNB, grouped by id",GaussianNB(),features),
                          ("GNB, grouped, no id/partner",GaussianNB(),no_id)):
        p = cross_val_predict(model(est),speedDating[cols],y,cv=kf,groups=groups)
        print("%-28s acc %.3f  prec %.3f  rec %.3f  f1 %.3f" % (name,accuracy_score(y,p),precision_score(y,p),recall_score(y,p),f1_score(y,p)))

def main():
    speedDating = pd.read_csv(ROOT / "speedDating_trab.csv")
    
    remove_columns(speedDating)
    remove_row(speedDating)
    
    my_list = list(speedDating)

    calc_mean_median_mode(my_list,speedDating,"mean")

    X = speedDating[["id","partner","age","age_o","goal","date","go_out","int_corr","length","met","like","prob"]]
    y = speedDating["match"]
    
    balanced(speedDating)
    
    print("CLF Holdout")
    clf = model(tree.DecisionTreeClassifier(random_state=42))
    clf_holdout(X,y,clf,0.3)
    export_tree("holdout",6,clf)

    print("\nCLF Cross Validation")
    clf = model(tree.DecisionTreeClassifier(random_state=42))
    clf_cross_validation(X,y,clf,5)
    export_tree("cv",6,clf)

    print("\nGNB Holdout")
    gnb = model(GaussianNB())
    gnb_holdout(X,y,gnb,0.3)
    
    print("\nGNB Cross Validation")
    gnb = model(GaussianNB())
    gnb_cv(X,y,gnb,5)

    print("\nExtra experiments (5-fold CV)")
    extra_experiments(speedDating,list(X.columns))
    print("\nGrouped CV (by id)")
    grouped_experiments(speedDating,list(X.columns))

if __name__ == "__main__":
    main()
