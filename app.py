import io
from flask import Flask, request, jsonify
from PIL import Image
from model_vit import result_vit
from model_bert import result_bert
from model_cat import result_cat

app = Flask(__name__)


@app.route("/predict_bert", methods=['POST'])
def predict_bert():
    """
    Ответ сервера по пути "/predict_bert".
    Функция принимает на вход изображение, с помощью функции ocr получает обработанный текст, токенизирует его,
    классифицирует с помощью Bert и возвращает текст и результаты классификации в виде вероятностей класса.
    """
    img_bytes = request.get_data()
    img = Image.open(io.BytesIO(img_bytes))
    probs = result_bert(img)

    return jsonify({"text": probs[0],
                    "normal": round(probs[1].item(), 4), "fraud": round(probs[2].item(), 4)})


@app.route("/predict_vit", methods=['POST'])
def predict_vit():
    """
    Ответ сервера по пути "/predict_vit".
    Функция принимает на вход изображение, конвертирует его, затем пропускает через список трансформаций,
    классифицирует его и возвращает результаты в виде вероятностей класса.
    """
    img_bytes = request.get_data()
    img = Image.open(io.BytesIO(img_bytes))
    probs = result_vit(img)

    return jsonify({"normal": round(probs[0].item(), 4), "fraud": round(probs[1].item(), 4)})


@app.route("/predict_pipe", methods=['POST'])
def predict_pipe():
    """
    Ответ сервера по пути "/predict_pipe".
    Функция принимает на вход изображение, предсказывает вероятности принадлежности к 1 классу с помощью ViT и Bert
    и на их основе с помощью CatBoost возвращает результаты классификации в виде вероятностей класса и предсказанный
    класс с порогом 0.5.
    """
    img_bytes = request.get_data()
    img = Image.open(io.BytesIO(img_bytes))
    probs = result_cat(img)

    return jsonify({"normal": round(probs[0].item(), 4),
                    "fraud": round(probs[1].item(), 4), "verdict": probs[2].item()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
