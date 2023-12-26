from flask import Flask
from flask_restful import Resource, Api
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

# Load models (assuming scaler is also a pickle file)
model = pickle.load(open('path_to_model.pkl', 'rb'))
scaler = pickle.load(open('path_to_scaler.pkl', 'rb'))

class Prediction(Resource):
    def get(self, feature1, feature2):
        f1, f2 = [feature1], [feature2]
        scaled_features = scaler.transform([[f1, f2]])
        df = pd.DataFrame({'feature1': f1, 'feature2': f2})
        prediction = model.predict(df)
        return str(int(prediction[0]))

class GetData(Resource):
    def get(self):
        df = pd.read_excel('data.xlsx')
        df = df.rename({'Feature 1': 'feature1', 'Feature 2':'feature2'}, axis=1)
        return df.to_json(orient='records')

api.add_resource(GetData, '/api')
api.add_resource(Prediction, '/prediction/<float:feature1>/<float:feature2>')

if __name__ == '__main__':
    app.run(debug=True)
