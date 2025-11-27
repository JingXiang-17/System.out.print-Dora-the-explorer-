# Data Source Documentation

## 1. Raw Dataset

- **Title:** Flight Delay 2020-2025 (Raw)
- **Location:** [Kaggle Dataset - Flight Delay 2020-2025 (CSV)](https://www.kaggle.com/datasets/jonathansia/flight-delay-2020-2025)
- **Format:** CSV
- **Description:**  
  Original, untouched airline flight delay data for the period 2020 to 2025, including all features downloaded from the U.S. Bureau of Transportation Statistics (BTS) via [FAA's Airline Time Performance and Causes of Flight Delays](https://www.bts.gov/explore-topics-and-geography/topics/airline-time-performance-and-causes-flight-delays).

## 2. Preprocessed, Train-Test Split Dataset

- **Title:** Train-Test Ready Dataset (Flight Delay)
- **Location:** [Kaggle Dataset - Train-Test Ready Dataset (Parquet)](https://www.kaggle.com/datasets/jonathansia/train-test-ready-dataset-flight-delay)
- **Format:** Parquet
- **Description:**  
  This dataset is derived from the raw flight delay dataset. It has been preprocessed (cleaned, encoded, and formatted), and split into train and test sets, stored in efficient Parquet format for direct use in model training.

---

## Data Preparation Steps (Summary)
- Download the raw dataset from the Kaggle link above or directly from BTS.
- Perform preprocessing: cleaning, handling missing values, feature engineering, encoding categorical features.
- Split into train/test sets and store in Parquet for efficient loading into model pipelines.

## Usage Notes
- For optimal results in machine learning models, ensure categorical columns are properly typed (e.g., `category` dtype for pandas).
- Refer to the appropriate Kaggle dataset depending on whether you need the original raw dataset or the pre-split, ready-to-use dataset.

## License and Attribution
- **Primary Source:**  
  - U.S. Bureau of Transportation Statistics (BTS), Federal Aviation Administration (FAA)
  - [Airline Time Performance and Causes of Flight Delays](https://www.bts.gov/explore-topics-and-geography/topics/airline-time-performance-and-causes-flight-delays)
- **Kaggle Datasets:**  
  Datasets compiled and uploaded by [jonathansia](https://www.kaggle.com/jonathansia)

Data is publicly available for research and educational use; please verify licensing on both the BTS and Kaggle sites.

## Citation Example
> Source: U.S. Bureau of Transportation Statistics; Kaggle: "Flight Delay 2020-2025" by jonathansia (raw), "Train-Test Ready Dataset - Flight Delay" by jonathansia (preprocessed).