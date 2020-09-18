import logging
import os
import re
from abc import ABC

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
import json

# from ts.metrics.dimension import Dimension
logger = logging.getLogger(__name__)


def get_config(path):
    if path:
        with open(path, "r") as f:
            config = json.load(f)
        return config
    else:
        return {"model_name": "dslim/bert-base-NER"}


# Returns grouped_entities=True
class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """

    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self._batch_size = 0
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        self.metrics = ctx.metrics

        logger.info(f"Manifest: {self.manifest}")

        properties = ctx.system_properties
        self._batch_size = properties["batch_size"]

        logger.info(f"properties: {properties}")

        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        config = get_config(os.path.join(model_dir, "config.json"))
        model_name = config["model_name"]

        # Read model serialize/pt file
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            ignore_labels=[],
            grouped_entities=True,
            #             ignore_subwords=True,
            device=self.device.index,
        )

        logger.debug(
            "Transformer model from path {0} loaded successfully".format(model_dir)
        )

        self.initialized = True

    def preprocess(self, data):
        """Very basic preprocessing code - only tokenizes.
        Extend with your own preprocessing steps as needed.
        """
        logger.info(f"Received data: {data}")
        processed_sentences = []
        for d in data:
            text = d.get("data")
            if text is None:
                text = d.get("body")
            sentence = text.decode("utf-8")
            logger.info("Received text: '%s'", sentence)

            # Modify this with your preprocessing
            num_separated = [s.strip() for s in re.split("(\d+)", sentence)]
            digit_processed = " ".join(num_separated)
            processed_sentences.append(digit_processed)

        return processed_sentences

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit
        # its expected input format.
        ents = self.nlp(inputs)
        if len(inputs) == 1:
            ents = [ents]
        return ents

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here

        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:

        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
