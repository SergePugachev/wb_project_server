<h1 align="center">WBTECH Проект "Репутация пользователей"
<h2 align="center">Задача</h2>
Задача данного проекта состоит в разработке алгоритма, позволяющего отличить допустимые изображения от недопустимых. 
Пользователи имеют возможность загружать изображения в раздел "Отзывы", недопустимыми являются изображения которые используются в качестве рекламы товаров и услуг, 
как правило они содержат контактные данные: номера телефонов, ссылку на сайт, аккаунт телеграмм и т.д. 
Подобные случаи могут негативно сказываться на доверие других пользователей к отзывам, а также угрожают их безопасности. 
Отзывы это очень важная часть любого маркетплейса, согласно статистике, это около 50% потребляемого контента на Wildberries и 93% покупателей принимают решение о покупке после прочтения отзыва. 
Задача является важной и решение данной проблемы будет влиять на рост доверия покупателей к марткетплейсу, что в свою очередь будет фактором увелечения числа покупок совершенных на платформе.
<h4 align="left">Примеры допустимых изображений:</h4>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/121925.png">
<h4 align="left">Примеры недопустимых изображений:</h4>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/120622.png">
<h2 align="center">Решение</h2>
<h4 align="center">I. EDA и предобработка данных</h4>
На первых этапах работы по графику распределения среднего значения цвета изображений, было выявлено большое количество выбросов, при дальнейшем 
анализе изображений были установленно, что выбросы в распределении являются дубликатами недопустимых изображений. 
  <h3 align="center"></h3>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/color_distribution.png"> 
Для того чтобы их найти я использовал библиотеку imagededup, сначала переводил изображения в новое признаковое пространство с помощью свертки, а затем находил похожие, вычисляя косинусное сходство между векторами, диапазон сходства от 0.9 до 1. После обнаружения удалил из датасета все дубликаты.
  <h3 align="center"></h3>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/dup_examples.png"> 
  <h3 align="center"></h3>
Затем на датасете недопустимых изображений без найденных дубликатов с помощью ocr считал на них текст, собрал все слова в BOW и проанализировал его.
Создал список слов, которые являются самыми частыми для класса недопустимых и отобрал все изображения на которых не было обнаружено ни одного из них. После кластеризации этих объектов получилось найти группы
с ошибочной разметкой: в основном это товары с большим количеством текста на упаковке или изображения промо листовок, которые сопутствуют товарам.
  <h3 align="center"></h3>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/false_label.png"> 
  <h3 align="center"></h3>
<h4 align="center">II. Pipeline</h4>
Для своего решения я разделил датасет на тренировочный и валидационный (0.7 train, 0.3 validation), затем с помощью EasyOCR считал текст с каждого изображения, обучил на тренировочных данных предобученные ViT_L_16 и BERT, сделал предикты на валидации и обучил CatBoost на этих предиктах. На тестовом датасете также сделал предикты по двум нейросетям, а финальное предсказание было сделано с помощью CatBoost по предсказаниям нейросетей. 
  <h3 align="center"></h3>
<h4 align="left">Результаты ViT на train и val:</h4>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/vit_res.png"> 
  <h3 align="center"></h3>
  <h3 align="center"></h3>
<h4 align="left">Результаты Bert на val с порогом 0.9:</h4>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/bert_res.png"> 
  <h3 align="center"></h3>
  <h3 align="center"></h3>
<h4 align="left">f1 score всего pipeline на kaggle competition:</h4>
<img src="https://github.com/SergePugachev/wb_project/blob/master/images/pipe_res.png"> 
  <h3 align="center"></h3>
<h4 align="center">III. Production</h4>
На последнием этапе работы я реализовал сервис, который принимает пользовательские запросы с изображением и классифицирует его с помощью ViT_L_16 и BERT, а также всего pipeline. Весь проект есть в образе на dockerhub.
Для того чтобы воспользоваться сервисом необходимо:
<h5 align="left">1.Скачать образ sergeypugachevv/wb-project-server</h5>
<h5 align="left">2.Запустить его командой docker run --rm -it -p 8080:8080 sergeypugachevv/wb-project-server <br/>(потребуется несколько минут чтобы развернуть контейнер и запустить сервер)</h5>
<h5 align="left">3.Код для теста:</h5>

```python
import requests
import io
from PIL import Image


def image_to_byte(image):
    img_byte = io.BytesIO()
    image.save(img_byte, format=image.format)
    img_byte = img_byte.getvalue()
    return img_byte

urls = {
    'vit': "http://localhost:8080/predict_vit",
    'bert': "http://localhost:8080/predict_bert",
    'pipe': "http://localhost:8080/predict_pipe"
}

img = Image.open("test.jpg")  # указать путь к файлу
img_bytes = image_to_byte(img)
resp = requests.post(urls['pipe'], data=img_bytes).json()  # можно выбрать ViT, BERT или весь pipeline
print(resp)
```

