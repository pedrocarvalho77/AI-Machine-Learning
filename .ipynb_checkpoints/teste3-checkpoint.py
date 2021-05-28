from os import spawnlp
import pandas as pd

speedDating = pd.read_csv("~/Faculdade/3ano/2sem/IA/trabalho2/AI-Machine-Learning/speedDating_trab.csv")

#print(speedDating)

#print(speedDating.head(8))

#print(speedDating.dtypes)

#print(speedDating.info())

#partners = speedDating["partner"]
#print(partners.head())

#print(speedDating.shape)

#age_sex = speedDating[["age", "partner", "goal"]]

#print(age_sex.head())

#above_35 = speedDating[speedDating["age"]>35]

#print(above_35.head())

print(speedDating["age"]>35)