import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

path = 'credit_risk_dataset.csv'

df = pd.read_csv(path)

df = df.replace({'loan_grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7},
                'person_home_ownership': {'OWN': 1, 'RENT': 2, 'MORTGAGE': 3}, 
                'loan_intent': {'PERSONAL': 1,'EDUCATION': 2,'MEDICAL': 3,'VENTURE': 4,'HOMEIMPROVEMENT': 5,'DEBTCONSOLIDATION': 6}})


y = df['loan_status']
x = df.drop('loan_status', axis=1)


print(x.columns)
print(y)


