from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@app.route("/predict", methods=["POST"])
def predict():
    hours = request.json["hours"]
    prediction = model.predict([[hours]])
    return jsonify({"Result": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

    #docker build -t "folder name"
    #docker run -p 5000:5000 "foldername"
    #Invoke-RestMethod -Method Post -Uri "https://127.0.0.1:5000/predict"-ContentType "application/json" -Body '{"hours":4.0}'
