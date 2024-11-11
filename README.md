
# Airline Delay Prediction

This project aims to analyze and predict airline delays using PySpark for data processing and machine learning, Docker for containerization, and Jupyter for interactive data analysis.

## Project Structure
```
├── data/
│   ├── raw/             # Raw datasets (not included in the repository)
│   ├── processed/       # Processed data (not included in the repository)
├── notebooks/           # Jupyter Notebooks for data exploration
├── scripts/
│   ├── airplanes_data_pipe.py    # Data processing pipeline
│   ├── airplanes_ml_pipe.py      # Machine learning pipeline
├── docker-compose.yaml # Docker configuration for the project
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## Prerequisites

- Docker
- Docker Compose
- Apache Spark
- Python 3.8 or higher
- Jupyter Notebook


## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nimonkaranurag/airline-delay-prediction.git
   cd airline-delay-prediction
   ```

2. **Build and Start the Docker Containers**:
   This command will start the Spark master, Spark worker, and Jupyter notebook services.
   ```bash
   docker-compose up -d
   ```

3. **Access Jupyter Notebook**:
   Open your browser and navigate to:
   ```
   http://localhost:8888
   ```

4. **Data Preparation**:
   - Place your raw dataset (e.g., `2018.csv`) in the `data/raw/` directory.
   - Run the `airplanes_data_pipe.py` script to process the raw data:
     ```bash
     python scripts/airplanes_data_pipe.py
     ```

5. **Train the Model**:
   Run the `airplanes_ml_pipe.py` script to train and evaluate the machine learning model:
   ```bash
   python scripts/airplanes_ml_pipe.py
   ```


## Data Processing Pipeline

The `airplanes_data_pipe.py` script performs the following tasks:
- Loads the raw data and infers schema using Spark.
- Generates temporal features (e.g., `DEP_HOUR`, `DAY_OF_WEEK`, `IS_WEEKEND`).
- Calculates airline-specific features (e.g., `AVERAGE_CARRIER_DELAY`, `AVERAGE_ROUTE_DELAY`).
- Creates delay features (e.g., `DELAY_CAT`, `DELAY_RISK_SCORE`).
- Saves the processed data in the `data/processed/` directory.

## Machine Learning Pipeline

The `airplanes_ml_pipe.py` script performs the following tasks:
- Checks the quality of the dataset (counts null values and class distribution).
- Prepares features using `StringIndexer`, `OneHotEncoder`, and `StandardScaler`.
- Trains a `RandomForestClassifier` model.
- Evaluates the model using metrics like `f1`, `precision`, `recall`, and `accuracy`.
- Saves the trained model in the `data/models/` directory.


## Troubleshooting

- **Large Files Error**: If you encounter errors while pushing large files to GitHub, use Git LFS to track them:
  ```bash
  git lfs track "*.csv"
  ```

- **File Not Found**: Ensure the dataset is placed in the correct directory (`data/raw/`).
- **Out of Memory**: If you face memory issues, reduce the number of records being processed by modifying the `number_of_records` parameter in `airplanes_ml_pipe.py`.

## Logging

Logs are configured using Python's `logging` module and will be displayed in the console. To save logs to a file, modify the `basicConfig` in both Python scripts:
```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s", filename="project.log", filemode="w")
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- Apache Spark for distributed data processing.
- Bitnami for the Spark Docker image.
- Jupyter for interactive data analysis.
- Kaggle for the airline delay dataset.

## Author

- Anurag Nimonkar
