import regressionModel
import pandas as pd


print("Please Provide the Information For The Following")
import pandas as pd

sex = (input("Male/Female: ")).lower()
age = float(input("Age: "))
n_siblings_spouses = int(input("Number of siblings and spouses on board: "))
fare = float(input("Fare amount: "))
_class = (input("Seat Class [First/Second/Third]: ")).capitalize()
deck = (input("Deck [A/B/C/D/E/unknown]: ")).upper()
alone = (input("Alone in the ship (y/n): ")).lower()

data = pd.DataFrame(data =[[sex, age, n_siblings_spouses, fare, _class, deck, alone]], 
                    columns=["sex", "age", "n_siblings_spouses", "fare", "class", "deck", "alone"])



accuracy, probability, prediction = regressionModel.Reg(data)

print("The accuracy of the model is: ", accuracy)

probability = probability*100

print("Probability of Survival: ", probability)

if prediction == 0:
    print(f"You are less likely to survive with a {probability} percent probability of survival")
else:
    print(f"You are likely to survive with a {probability} probability of survival")

