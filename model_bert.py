import torch
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import torch.nn.functional as f
import easyocr
import transformers
from torch import nn, optim

nltk.download('stopwords')
punc = list(punctuation)
punc.remove('@')
punc.remove('$')
punc += [' ', '\n']  # Список пунктуации
stop_list = stopwords.words('russian')  # Список стоп слов

reader = easyocr.Reader(['ru', 'en'])
mystem = Mystem()


def ocr(img):
    """
    Функция принимает на вход изображение, с помощью EasyOCR считывает текст, затем лемматизирует его,
    убирает стоп слова, пунктуацию и возвращает получившийся текст
    """
    words = []
    list_text_from_img = reader.readtext(img, detail=0)
    text = " ".join(list_text_from_img)
    tokens = mystem.lemmatize(text.lower())
    for token in tokens:
        token = token.translate(str.maketrans('', '', ''.join(punc)))
        if token not in stop_list and len(token) > 0:
            words.append(token)
    words = " ".join(words)
    return words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_pretrained = transformers.AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence',
                                                         resume_download=None)
tokenizer = transformers.BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence',
                                                       resume_download=None)

for param in bert_pretrained.parameters():
    param.requires_grad = False


class BertArch(nn.Module):
    def __init__(self, bert):
        super(BertArch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model_bert = BertArch(bert_pretrained)
model_bert = model_bert.to(device)
optimizer = optim.Adam(model_bert.parameters(), lr=0.0003)
model_bert.load_state_dict(torch.load('bert_weights.pt', map_location=torch.device('cpu')))


def result_bert(img):
    """
    Функция принимает на вход изображение, с помощью функции ocr получает обработанный текст, токенизирует его,
    классифицирует с помощью Bert и возвращает текст и результаты классификации в виде вероятностей класса
    """
    review_text = ocr(img)
    tokens = tokenizer.encode_plus(
        review_text,
        max_length=50,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    pred_logits_tensor = model_bert(input_ids, attention_mask)
    pred_probs = f.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    normal, fraud = pred_probs[0][0], pred_probs[0][1]
    return review_text, normal, fraud
