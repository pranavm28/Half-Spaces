import gdown
import pandas as pd


def download_csv_and_convert_to_parquet(file_id: str, parquet_filename: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    csv_filename = "temp_download.csv"
    gdown.download(url, csv_filename, quiet=False)
    df = pd.read_csv(csv_filename)
    df.to_parquet(parquet_filename, index=False)
    return parquet_filename


# Example usage:
parquet_path = download_csv_and_convert_to_parquet(
    "1z_Bb2kf48NqwrRO_ZSZeJLhE9mJ1ps71", "Bundesliga_2425.parquet"
)
parquet_path = download_csv_and_convert_to_parquet(
    "1zeJtDsuTNwd3EKc9fY5FhNBdfl9sV9an", "La_Liga_24_25.parquet"
)
parquet_path = download_csv_and_convert_to_parquet(
    "154GIano_ASZIGf3cOayy7OCIF9NjJx2x", "Ligue_1_2425.parquet"
)
parquet_path = download_csv_and_convert_to_parquet(
    "1mCq0wlrlnohawuuYCZPTulvkQwqdeYt6", "Premier_League_2425.parquet"
)
parquet_path = download_csv_and_convert_to_parquet(
    "1OVkg5E2whpoE_snZT2fxkN_8mMgnSU0i", "Serie_A_2425.parquet"
)
