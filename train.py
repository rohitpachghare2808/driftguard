import json
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X, y = load_wine(as_frame=True, return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
f1 = f1_score(yte, model.predict(Xte), average="weighted")

joblib.dump(model, "model.pkl")
Xtr.to_csv("reference_data.csv", index=False)
json.dump({"f1": f1}, open("metrics.json", "w"))
print("F1:", f1)
