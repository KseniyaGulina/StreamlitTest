import streamlit as st
import requests
from PIL import Image

API_URL_DICT = {
    'afficient-b7': "https://api-inference.huggingface.co/models/google/efficientnet-b7",
    'resnet-18': "https://api-inference.huggingface.co/models/microsoft/resnet-18",
}

headers = {"Authorization": "Bearer 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"}


def query(model, data):
    response = requests.post(API_URL_DICT[model], headers=headers, data=data)
    return response.json()


def input_features():
    model = st.selectbox("Модель: ", API_URL_DICT.keys())
    return model


def predict(model, data):
    result = query(model, data)
    return result


def inference(model, upload):
    c1, c2 = st.columns(2)
    if upload is not None:
        output = predict(model, upload)
        im = Image.open(upload)
        c1.header("Imput Image")
        c1.image(im)
        c2.header("Predicted class: ")
        c2.write(output[0]['label'])


def show_main_page():
    st.set_page_config(page_title="Two Model Inference")
    st.title("Распознавание объекта")

    model = input_features()
    st.header("Введите картинку:")
    upload = st.file_uploader('Выберите картинку:', type=['png', 'jpg'])
    inference(model, upload)


if __name__ == '__main__':
    show_main_page()
