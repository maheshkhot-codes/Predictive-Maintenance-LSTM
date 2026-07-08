# Predictive Maintenance System using LSTM

An AI-powered predictive maintenance system for CNC machines that uses an LSTM (Long Short-Term Memory) model to analyze time-series sensor data and predict potential machine failures before they occur.

## Features

- LSTM-based machine failure prediction
- Real-time machine health monitoring
- Interactive Plotly dashboard
- Flask web application
- Time-series sensor data analysis

## Tech Stack

- Python
- TensorFlow / Keras
- Flask
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Project Structure

```
├── predictive_maintenance_cnc_flask_lstm.py
├── predictive_maintenance_cnc_flask_plotly_dark.py
├── train_lstm_cnc.py
├── lstm_cnc_model.h5
├── scaler_cnc.pkl
├── cnc_log_threshold_breaking.csv
├── .gitignore
└── README.md
```

## How to Run

```bash
git clone https://github.com/maheshkhot-codes/Predictive-Maintenance-LSTM.git
cd Predictive-Maintenance-LSTM
pip install -r requirements.txt
python predictive_maintenance_cnc_flask_lstm.py
```

## Future Improvements

- Live IoT sensor integration
- Email and SMS alerts
- Cloud deployment
- Multi-machine monitoring

## Author
**Mahesh Khot**
