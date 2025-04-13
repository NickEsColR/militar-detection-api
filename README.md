# Military Aircraft Detection API

## Overview

This project provides a backend API for detecting military aircraft in images. It leverages a deep learning model built with PyTorch, offering an open-weight architecture for transparency and further development. The model uses SqueezeNet as a backbone and RetinaNet-based heads for accurate object detection.  The API itself is constructed using FastAPI, ensuring efficient and robust performance.

## Table of Contents

* [Features](#features)
* [Technologies](#technologies)
* [How to Run](#how-to-run)
* [Acknowledgements](#acknowledgements)
* [Authors](#authors)

## Features

* **Military Aircraft Detection:**  Identifies and localizes military aircraft within images.
* **Open-Weight Model:** The underlying PyTorch model and weights are publicly available, allowing for inspection, modification, and fine-tuning.
* **FastAPI Backend:** Provides a high-performance and easy-to-use API for interacting with the model.

## Technologies

* **PyTorch:** Deep learning framework for building and training the detection model.
* **FastAPI:** Modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.
* **Python:**  Primary programming language.

## How to Run

For development run

```bash
source .venv/bin/activate
fastapi dev main.py
```

For production run

```bash
source .venv/bin/activate
fastapi run main.py
```

## Acknowledgements

This project continues the competition for the master "Applied Artificial Intelligence" [AAIV 2025-I:Object Location](https://www.kaggle.com/competitions/aa-iv-2025-i-object-localization)

This project utilizes several open-source resources and libraries. We would like to acknowledge the contributions of the PyTorch, SqueezeNet, RetinaNet, and FastAPI communities.

## Authors

[NickEsColR](https://github.com/NickEsColR) | [Carlos Martinez](https://github.com/cam2149)
