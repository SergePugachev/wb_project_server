import requests
from urllib.parse import urlencode

public_keys = ['https://disk.yandex.ru/d/67Rve_2BDr-2qg', 'https://disk.yandex.ru/d/tIHvamBkNOikAw',
               'https://disk.yandex.ru/d/RI6wTCijVZ1gUA']
file_names = ['vit_weights.pt', 'cat_weights.cbm', 'bert_weights.pt']
base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

for i in range(3):
    # Загружаем веса для трех моделей, которые находятся на яндекс диске
    final_url = base_url + urlencode(dict(public_key=public_keys[i]))  # Получаем загрузочную ссылку
    response = requests.get(final_url)
    download_url = response.json()['href']
    download_response = requests.get(download_url)
    with open(file_names[i], 'wb') as file:
        file.write(download_response.content)
