# Wine Predictor

Wine Predictor is an AI applet that anyone can use to predict the rating of a wine if they have the characteristic data for the wine as it was defined in the training data.
The predictor uses a trained sequential neural network model to make the predictions. The UI was created using streamlit

## Environments
### venv

Use [venv](https://docs.python.org/3/library/venv.html) to launch a python environment 3.8 or higher

```bash
source venv/bin/activate
```

### conda

or use [conda](https://docs.conda.io/en/latest/) to launch a python environment 3.8 or higher

```bash
source venv/bin/activate
```

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install tensorflow
pip install streamlit
```
## Running the App
### using terminal

```bash
#run python env locally
cd winepredictor
python main.py
````
#### after main.py application has compiled run:
```bash
streamlit run main.py 
````

```bash
You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8502
  Network URL: http://192.168.1.153:8502
````
### using an IDE:
```python
import streamlit as st
import tensorflow as tf
from pathlib import Path
import pickle
```
```bash
run main.py
```

#### streamlit will launch automatically. view the app in the browser:
```bash
http://localhost:8501/
````


## Usage

> Enter in the attributes on the side panel and hit the predict button to get a predicted quality

**characteristics:**
- fixed_acidity
- volatile_acidity
- citric_acidity
- residue_sugar
- chlorides
- free_sulfur_dioxide
- total_sulfur_dioxide
- density
- ph
- sulphates
- alcohol


## Re-Usage
### To create and train a new model

#### change training data
Open file: `model_train_save.py` and update path to new csv

```python
data_training_file_path = "training_data/wine_quality.csv"
````

```bash
#### saved models:
models/wine_quality.h5
```

## Contributing

Pull requests are welcomed but please open an issue first
to discuss what you would like to add or change. This is mostly for demo purposes to show how to create, train and save a sequential neural network, reload it and use it in a streamlit application