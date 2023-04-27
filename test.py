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

print(data.head())
print(data.info())