import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("Crop_recommendation.csv")

x = data.iloc[: ,:-1]
y  = data.iloc[: ,-1]


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

model = RandomForestClassifier()

model.fit(x_train , y_train)

prediction = model.predict(x_test)

accuracy = model.score(x_test , y_test)

print(accuracy)

import pickle
pickle.dump(model, open('crop_recommendation_model.pkl', 'wb'))
filename = 'crop_recommendation_model.pkl'