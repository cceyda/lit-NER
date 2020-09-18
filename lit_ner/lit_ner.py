import os

import streamlit as st
from spacy import displacy

import httpx
from utils import hf_ents_to_displacy_format, make_color_palette

# Modify these
API_URL = "http://127.0.0.1:7863/predictions/"
MODEL_NAME = "ner_model"

# from https://github.com/explosion/spacy-streamlit/util.py#L26
WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache()
def get_colormap():
    with open("./labels.txt", "r") as f:
        labels = f.read().splitlines()
    color_map = make_color_palette(labels)
    return color_map


def sanitize_input(input_):
    clean = str(input_)
    return clean


def predict(model, input_):
    res = httpx.post(API_URL + model, data=input_)
    return res.json()


def display(bert_ents):
    bert_doc = hf_ents_to_displacy_format(bert_ents, ignore_entities=["O"])
    html = displacy.render(
        bert_doc, manual=True, style="ent", options={"colors": color_map}
    )

    html = html.replace("\n", " ")
    st.write(WRAPPER.format(html), unsafe_allow_html=True)


color_map = get_colormap()

st.header("NER")
input_ = st.text_input("Input", "My name is Ceyda and I live in Seoul, Korea.")
input_ = sanitize_input(input_)
bert_ents = predict(MODEL_NAME, input_)
if bert_ents:
    display(bert_ents)
    st.write(bert_ents)
