import os

import streamlit as st
from spacy import displacy

import httpx
from lit_ner.utils import hf_ents_to_displacy_format, make_color_palette
from httpx import HTTPError
import random

# Modify these
API_URL = "http://127.0.0.1:7863/predictions/"
MODEL_NAME = "ner_model"
LOCAL = False

# from https://github.com/explosion/spacy-streamlit/util.py#L26
WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

if not LOCAL:
    API_URL = "https://api-inference.huggingface.co/models/"
    MODEL_NAME = "dslim/bert-base-NER"
    API_URL = st.sidebar.text_input("API URL", API_URL)
    MODEL_NAME = st.sidebar.text_input("MODEL NAME", MODEL_NAME)
    st.sidebar.markdown("Related blog [post](https://cceyda.github.io/blog/huggingface/torchserve/streamlit/ner/2020/10/09/huggingface_streamlit_serve.html)")
    st.write(f"API endpoint: {API_URL}{MODEL_NAME}")


def raise_on_not200(response):
    if response.status_code != 200:
        st.write("There was an error!")
        st.write(response)


client = httpx.Client(timeout=1000, event_hooks={"response": [raise_on_not200]})


@st.cache(allow_output_mutation=True)
def get_colormap():
    with open("./examples/labels.txt", "r") as f:
        labels = f.read().splitlines()
    color_map = make_color_palette(labels)
    return color_map


def add_colormap(labels):
    for label in labels:
        if label not in color_map:
            rand_color = "#"+"%06x" % random.randint(0, 0xFFFFFF)
            color_map[label]=rand_color
    return color_map


def sanitize_input(input_):
    clean = str(input_)
    return clean


def predict(model, input_):
    res = client.post(API_URL + model, json=input_)
    return res.json()


def display(bert_ents):
    bert_doc = hf_ents_to_displacy_format(bert_ents, ignore_entities=["O"])
    labels = list(set([a["label"] for a in bert_doc["ents"]]))
    color_map = add_colormap(labels)
    html = displacy.render(bert_doc, manual=True, style="ent", options={"colors": color_map})

    html = html.replace("\n", " ")
    st.write(WRAPPER.format(html), unsafe_allow_html=True)


color_map = get_colormap()

st.header("NER")
input_ = st.text_input("Input", "My name is Ceyda and I live in Seoul, Korea.")
input_ = sanitize_input(input_)
bert_ents = predict(MODEL_NAME, input_)
if bert_ents:
    if isinstance(bert_ents, dict) and "error" in bert_ents:
        st.write(bert_ents)
    else:
        display(bert_ents)
        st.write(bert_ents)
