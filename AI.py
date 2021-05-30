import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from IPython.display import Image
import pydotplus

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

def calc_mean_median_mode():
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            print("Mean",i,"=",speedDating[i].mean())
            print("Median",i,"=",speedDating[i].median())
            print("Mode",i,":")
            print(speedDating[i].mode())

calc_mean_median_mode()

def replace_missing_values():
    for i in my_list:
        if i != 'Unnamed: 0' and i != 'match':
            speedDating[i] = speedDating[i].fillna(speedDating[i].mean())

replace_missing_values()

features_names2= ['id','partner','age','age_o','goal','date','go_out','int_corr','length','met','like','prob']
target_names = ['0','1']
X = speedDating[["id","partner","age","age_o","goal","date","go_out","int_corr","length","met","like","prob"]]

y = speedDating["match"]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

print(clf.get_depth())

print(clf.predict([[5,11,15,13,3,3,2,0.7,3,0,8,1]]))
print(clf.predict([[5,11,15,13,3,3,2,0.7,3,0,3,1]]))

print(clf.predict_proba([[5,11,15,13,3,3,2,0.7,3,0,8,1]]))
print(clf.predict_proba([[5,11,15,13,3,3,2,0.7,3,0,3,1]]))

tree.plot_tree(clf)
dot = tree.export_graphviz(clf,max_depth=3,out_file=None,feature_names=features_names2,class_names=target_names)
#dot = tree.export_graphviz(clf,out_file=None,feature_names=features_names2,class_names=target_names)
graph = pydotplus.graph_from_dot_data(dot)
Image(graph.create_png())
graph.write_png("tree_depth3.png")
#graph.write_png("tree_no_depth_limit.png")
