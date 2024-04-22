# Handwritten digit recognition

This repository can be used for educational purpose by anyone interested in learning about neural networks.

## Prerequisites

- Verify Python version
  ```bash
  python --version # Python 3.12.2
  ```
- Create virtual environment
  ```bash
  python -m venv .venv
  ```
- Activate virtual environment
  ```bash
  source .venv/bin/activate
  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

## Training a model

[model.json](model.json) has been trained based on [MNIST dataset](http://yann.lecun.com/exdb/mnist/) by running:

```bash
python trainer.py
```

## Running the demo

Showcase the model in a gradio-based demo by running:

```bash
python app.py
```

## References

- [YouTube playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=RFTfo6mK-fBgvJgQ) by 3Blue1Brown
- [GitHub repo](https://github.com/mnielsen/neural-networks-and-deep-learning) by mnielsen
