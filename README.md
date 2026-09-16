# Anomaly Detection in IoT Data

A practical deep-learning project for detecting unusual IoT sensor behavior using an autoencoder built with Keras. The model learns normal time-series patterns and identifies unusual readings using reconstruction error and a learned anomaly threshold.

The project uses the open-source **Numenta Anomaly Benchmark (NAB)** dataset and includes Jupyter notebooks for exploration, model training, and evaluation, along with a Flask web application for demonstrating anomaly detection interactively.

---

## Project Overview

IoT devices continuously generate sensor measurements such as temperature, pressure, vibration, voltage, and other operational signals.

Most sensor readings represent normal operating conditions, but unusual patterns can indicate:

* Sensor malfunction
* Equipment problems
* Unexpected operating conditions
* Sudden changes in system behavior
* Potential maintenance requirements

Instead of training a traditional classification model, this project uses an **autoencoder** to learn how normal sensor sequences look.

When the trained model receives a sequence that differs significantly from the patterns it learned, the reconstruction error becomes higher. A threshold is then used to classify the sequence as either **Normal** or **Anomaly**.

---

## Dataset

This project uses the **Numenta Anomaly Benchmark (NAB)**.

The notebooks work with:

* `art_daily_small_noise.csv` — normal sensor behavior used for training
* `art_daily_jumpsup.csv` — sensor behavior containing anomalous patterns for evaluation

Dataset repository:

https://github.com/numenta/NAB

The notebooks download the required dataset directly from the NAB repository, so large source datasets do not need to be committed to this repository.

---

## Skills Demonstrated

* Time-series analysis
* Exploratory data analysis
* Data normalization
* Sliding-window sequence generation
* Autoencoders
* Deep learning
* Keras
* TensorFlow
* Reconstruction error
* Threshold-based anomaly detection
* Model evaluation
* Data visualization
* Flask deployment
* HTML/CSS
* Machine learning model serving

---

## How the Project Works

```text
IoT Sensor Readings
        │
        ▼
Data Exploration
        │
        ▼
Normalization
        │
        ▼
Sliding Time Windows
        │
        ▼
Autoencoder
   ┌────┴────┐
   ▼         ▼
Encoder    Decoder
   │         │
   └────┬────┘
        ▼
Reconstructed Sequence
        │
        ▼
Reconstruction Error
        │
        ▼
Compare With Threshold
      /       \
     /         \
    ▼           ▼
 NORMAL       ANOMALY
```

The autoencoder learns to reconstruct normal sensor sequences.

For a new sequence:

```text
Reconstruction Error <= Threshold
              ↓
           NORMAL
```

or:

```text
Reconstruction Error > Threshold
              ↓
           ANOMALY
```

---

## Repository Structure

```text
Anomaly-Detection-in-IoT-Data/
│
├── data/
│   └── README.md
│
├── models/
│   ├── iot_autoencoder.keras
│   ├── README.md
│   ├── scaler.json
│   └── threshold.json
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Autoencoder_Training.ipynb
│   └── 03_Anomaly_Evaluation.ipynb
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── CHANGELOG.md
├── CONTRIBUTE.md
├── LICENSE
├── README.md
└── requirements.txt
```

### Important Repository Design

This project intentionally keeps the repository simple.

There is:

* No `src/` directory
* No separate preprocessing module
* No data-loader module
* No utilities module
* No model-helper module
* No unnecessary additional Python scripts

The main data science workflow is kept inside the notebooks, while the Flask application is contained in the single `app.py` file.

---

## Model Files

The `models/` directory contains the trained artifacts required by the web application.

### `iot_autoencoder.keras`

The trained Keras autoencoder used to reconstruct sensor sequences.

### `scaler.json`

Contains the scaling parameters required to transform incoming sensor readings in the same way as the training data.

### `threshold.json`

Contains the reconstruction-error threshold used to determine whether a sensor sequence should be classified as normal or anomalous.

### `models/README.md`

Documents the purpose of the trained model artifacts and how they are used by the Flask application.

---

## Project Notebooks

### 1. Data Exploration

```text
notebooks/01_Data_Exploration.ipynb
```

This notebook:

* Downloads the NAB dataset
* Loads the sensor data
* Converts timestamps
* Checks the data
* Visualizes sensor readings
* Examines normal and unusual patterns
* Prepares the data for model training

---

### 2. Autoencoder Training

```text
notebooks/02_Autoencoder_Training.ipynb
```

This notebook:

* Normalizes sensor values
* Creates sliding time windows
* Builds the Keras autoencoder
* Trains the model on normal sensor behavior
* Calculates reconstruction errors
* Determines an anomaly threshold
* Saves the trained model
* Saves the scaler parameters
* Saves the detection threshold

Generated files:

```text
models/iot_autoencoder.keras
models/scaler.json
models/threshold.json
```

---

### 3. Anomaly Evaluation

```text
notebooks/03_Anomaly_Evaluation.ipynb
```

This notebook:

* Loads the trained model
* Loads the scaling parameters
* Loads the anomaly threshold
* Generates predictions
* Calculates reconstruction errors
* Detects anomalous windows
* Visualizes anomaly scores
* Compares normal and unusual sensor behavior

---

# Installation

## Requirements

Python **3.12** is recommended for this project.

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in this order:

```text
01_Data_Exploration.ipynb
        ↓
02_Autoencoder_Training.ipynb
        ↓
03_Anomaly_Evaluation.ipynb
```

The training notebook should be completed before using the Flask application.

---

# Running the Web Application

The project includes a Flask web application so the trained model can be demonstrated outside the notebooks.

Run:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000
```

The web application allows a user to provide sensor readings and receive an anomaly detection result.

The result includes information such as:

* Prediction status
* Reconstruction error
* Detection threshold
* Normal/anomaly classification
* Explanation of the result

---

# Web Application

The Flask interface is designed as a simple demonstration of how a trained anomaly detection model can be integrated into an application.

```text
User Input
    ↓
Flask Application
    ↓
Load Scaler
    ↓
Normalize Input
    ↓
Load Autoencoder
    ↓
Generate Reconstruction
    ↓
Calculate Reconstruction Error
    ↓
Compare With Threshold
    ↓
Display Result
```

The web application also provides links to my professional profiles.

### GitHub

https://github.com/InfinitePraveen

### LinkedIn

https://www.linkedin.com/in/infinitepraveen/

---

# Interview Demonstration

This project can be demonstrated in an interview using the following flow:

### Step 1 — Explain the Problem

Explain why unusual IoT sensor behavior needs to be detected automatically.

### Step 2 — Show the Dataset

Open:

```text
01_Data_Exploration.ipynb
```

Show the normal and anomalous sensor patterns.

### Step 3 — Explain the Autoencoder

Explain the basic architecture:

```text
Input
  ↓
Encoder
  ↓
Latent Representation
  ↓
Decoder
  ↓
Reconstructed Input
```

### Step 4 — Explain Reconstruction Error

The model attempts to reconstruct the input sequence.

If the sequence resembles normal training data, reconstruction error should generally remain relatively low.

If the sequence is significantly different, reconstruction error can increase.

### Step 5 — Explain the Threshold

A threshold is used to convert the continuous reconstruction error into an anomaly decision.

### Step 6 — Show the Evaluation Notebook

Open:

```text
03_Anomaly_Evaluation.ipynb
```

Show the anomaly scores and detected unusual windows.

### Step 7 — Demonstrate the Flask Application

Start:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Enter sensor values and demonstrate how the application produces the anomaly result.

---

# Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Programming language                  |
| Pandas           | Data manipulation                     |
| NumPy            | Numerical operations                  |
| Matplotlib       | Visualization                         |
| Scikit-learn     | Data scaling and evaluation utilities |
| TensorFlow       | Deep learning framework               |
| Keras            | Autoencoder implementation            |
| Flask            | Web application                       |
| HTML             | Web interface                         |
| CSS              | Web interface styling                 |
| Jupyter Notebook | Data science workflow                 |

---

# Anomaly Detection Approach

The project uses **reconstruction-based anomaly detection**.

The autoencoder is trained to reproduce normal sensor sequences.

For every input sequence:

```text
Reconstruction Error =
Difference between original and reconstructed sequence
```

The resulting error acts as an anomaly score.

A simplified decision rule is:

```text
if reconstruction_error > threshold:
    anomaly
else:
    normal
```

This approach is useful when labeled examples of every possible anomaly are not available.

---

# Advantages of the Approach

* Learns patterns from normal sensor behavior
* Does not require a traditional multi-class anomaly classifier
* Works naturally with time-series windows
* Produces a continuous anomaly score
* Can be integrated into an application
* Provides a practical example of unsupervised/semi-supervised anomaly detection

---

# Limitations

The model should not be treated as a complete production monitoring system without additional validation.

Potential limitations include:

* The model depends on the quality of the normal training data.
* The anomaly threshold can affect detection results.
* A change in normal operating conditions may produce higher reconstruction error.
* Different IoT sensors may require different models and scaling strategies.
* Real production systems may require continuous monitoring and retraining.

---

# Future Improvements

Possible extensions include:

* Real-time sensor streaming
* MQTT integration
* Multiple IoT sensor inputs
* LSTM autoencoders
* Variational autoencoders
* Real-time anomaly dashboards
* Database integration
* Alert notifications
* Docker deployment
* Cloud deployment
* Model monitoring
* Automatic model retraining

---

# Contribution

Contributions are welcome.

Please read:

```text
CONTRIBUTE.md
```

before making changes to the project.

---

# Changelog

Project changes are documented in:

```text
CHANGELOG.md
```

---

# License

This project is released under the **MIT License**.

The NAB dataset remains subject to its own project terms. Please refer to the NAB repository for dataset licensing and usage information.

---

# Author

## Praveen Kumar

**Data Scientist | Open Source Learner | IBM Certified**

GitHub:
https://github.com/InfinitePraveen

LinkedIn:
https://www.linkedin.com/in/infinitepraveen/

---

## Project Purpose

This project was created as a practical demonstration of applying deep learning to time-series anomaly detection and deploying the resulting model through a simple web application.
