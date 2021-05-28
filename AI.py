import pandas as pd
from sklearn import tree
import graphviz 

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

#print(speedDating.head(10))
#print(speedDating.tail(2))

x = speedDating[["id","partner","age","age_o","goal","date","go_out","int_corr","length","met","like","prob"]]
#print(x)
y = speedDating["match"]
print(y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

print(clf.predict([[5,11,15,13,3,3,2,0.7,3,0,8,1]]))
print(clf.predict([[5,11,15,13,3,3,2,0.7,3,0,3,1]]))

print(clf.predict_proba([[5,11,15,13,3,3,2,0.7,3,0,8,1]]))
print(clf.predict_proba([[5,11,15,13,3,3,2,0.7,3,0,3,1]]))

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("speedDating")
#tree.plot_tree(clf)