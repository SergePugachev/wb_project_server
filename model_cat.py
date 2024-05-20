from model_vit import result_vit
from model_bert import result_bert
import numpy as np
from catboost import CatBoostClassifier

model_cat = CatBoostClassifier()
model_cat.load_model('cat_weights.cbm', format='cbm')


def result_cat(img):
    """
    Функция принимает на вход изображение, предсказывает вероятности принадлежности к 1 классу с помощью ViT и Bert
    и на их основе с помощью CatBoost возвращает результаты классификации в виде вероятностей класса и предсказанный
    класс с порогом 0.5
    """
    _, fraud_vit = result_vit(img)
    text, _, fraud_bert = result_bert(img)
    probs = np.array([fraud_bert, fraud_vit])
    preds_proba = model_cat.predict_proba(probs)
    preds_class = model_cat.predict(probs, prediction_type='Class')
    normal, fraud = preds_proba[0], preds_proba[1]
    return normal, fraud, preds_class
