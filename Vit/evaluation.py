    import pandas as pd
import os
import argparse

def calculate_accuracy(test_csv_path, prediction_csv_path):
    # Load test and prediction CSV files
    test_df = pd.read_csv(test_csv_path)
    prediction_df = pd.read_csv(prediction_csv_path)

    # Extract only filenames from the Filepath column
    test_df["Filename"] = test_df["Filepath"].apply(lambda x: os.path.basename(x))
    prediction_df["Filename"] = prediction_df["Filepath"].apply(lambda x: os.path.basename(x))

    # Merge on the Filename column to align targets with predictions
    merged_df = pd.merge(test_df, prediction_df, on="Filename", suffixes=('_test', '_pred'), how="inner")
    
    # Compare the target and predicted columns (case-insensitive comparison)
    correct_predictions = (merged_df["Language_test"].str.lower() == merged_df["Language_pred"].str.lower()).sum()
    total_predictions = len(merged_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

# If this file is run directly, accept inputs
if __name__ == "__main__":


    # Argument parser to take inputs from the command line
    parser = argparse.ArgumentParser(description="Calculate accuracy by comparing predictions with targets")
    parser.add_argument("test_csv_path", type=str, help="Path to the test CSV file with image locations and targets")
    parser.add_argument("prediction_csv_path", type=str, help="Path to the prediction CSV file with image locations and predictions")

    args = parser.parse_args()

    # Run the accuracy calculation
    calculate_accuracy(args.test_csv_path, args.prediction_csv_path)
