=======
lit-NER
=======

Streamlit demo for HuggingFace NER models

*****
Requirements
*****

spacy>=2.3.2

streamlit>=0.62.1

transformers>=3.1.0

torchserve_

.. _torchserve: http://pytorch.org/serve/install.html

(might also work with lower versions...not tested)

*****
How To
*****

:code:`git clone https://github.com/cceyda/lit-NER.git`

Serve a model using torchserve
########

If you don't have a NER model use :code:`examples/serve_pretrained.ipynb`
 
OR

If you have a pretrained model use :code:`examples/serve.ipynb`
 
Start the Streamlit Demo 
########

:code:`examples/start_demo.ipynb`


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* Free software: Apache Software License 2.0

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
