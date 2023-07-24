# from cellSegmentation.logger import logging
# from cellSegmentation.exception import AppException
# import sys
#
# from cellSegmentation.pipeline.training_pipeline import TrainingPipeline
#
# obj = TrainingPipeline()
# obj.run_pipeline()
# print("Training Done")


import sys,os
from cellSegmentation.pipeline.training_pipeline import TrainingPipeline
from cellSegmentation.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from cellSegmentation.constant.application import APP_HOST, APP_PORT
import mlflow

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/train", methods=['GET'])
def trainRoute():
    model_type = request.args.get('model_type')
    obj = TrainingPipeline()
    obj.run_pipeline(model_type=model_type)
    return "Training Successfull!!"


@app.route("/yolo")
def home_yolo():
    return render_template("index-yolo.html")


@app.route("/tf")
def home_tf():
    return render_template("index-tf.html")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict/tf", methods=['POST','GET'])
@cross_origin()
def predictRoute_tf():
    try:
        result =  "No Prediction/Training for Tensorflow!!"
    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


@app.route("/predict/yolo", methods=['POST','GET'])
@cross_origin()
def predictRoute_yolo():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        experiment_name = "ultralytics/yolov8"
        current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id = current_experiment['experiment_id']
        df = mlflow.search_runs([experiment_id], order_by=["metrics.rmse DESC"])
        best_run_id = df.loc[0, 'run_id']
        mlflow.artifacts.download_artifacts(run_id=best_run_id, artifact_path='best.pt',
                                            dst_path=os.path.join(os.getcwd(), 'artifacts/model_trainer'))

        os.system("yolo task=detect mode=predict model=artifacts/model_trainer/best.pt conf=0.25 source=data/inputImage.jpg save=true")

        opencodedbase64 = encodeImageIntoBase64("runs/segment/predict/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf runs")
        os.system("rm -rf data")

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)
    # app.run(host='0.0.0.0', port=80) #for AZURE
