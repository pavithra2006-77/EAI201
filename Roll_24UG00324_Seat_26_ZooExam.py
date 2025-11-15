import pandas as pd
import os

def Gamma_load_and_integrate():
    """
    Load 'class.csv', 'zoo.csv', and 'auxiliary_metadata.json' from 'lab-exam-data',
    integrate them into a single DataFrame, normalize names, fix JSON inconsistencies,
    merge datasets, and handle missing auxiliary data based on roll number.
    """

    folder_path = "lab-exam-data"  # Folder where your datasets are

    csv_files = ["class.csv", "zoo.csv"]
    json_file = "auxiliary_metadata.json"

    dataframes = []

    # ----- A: Data Loading -----
    for f in csv_files:
        file_path = os.path.join(folder_path, f)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)  # reset index
            dataframes.append(df)
        else:
            print(f"File not found: {file_path}")

    json_path = os.path.join(folder_path, json_file)
    if os.path.exists(json_path):
        df_json = pd.read_json(json_path)
        df_json.reset_index(drop=True, inplace=True)
        # ----- C: Fix JSON inconsistencies -----
        # Standardize column names
        df_json.rename(columns={
            "staNADARIZE": "conversation_status",
            "habitat_type": "habitat_type",
            "diet_type": "diet_type"
        }, inplace=True)

        # Fix diet typos
        df_json['diet_type'] = df_json['diet_type'].replace({
            "omnivor": "Omnivore",
            "fresh waterr": "Freshwater",
            "herbivor": "Herbivore",
            "carnivor": "Carnivore"
        })

        # Optional: title-case habitat_type
        if 'habitat_type' in df_json.columns:
            df_json['habitat_type'] = df_json['habitat_type'].astype(str).str.title()

        dataframes.append(df_json)
    else:
        print(f"File not found: {json_path}")

    # ----- D: Merge all datasets safely -----
    # Ensure all columns are unique and reset indexes
    for i, df in enumerate(dataframes):
        df.columns = [f"{col}" for col in df.columns]  # just in case duplicate columns
        df.reset_index(drop=True, inplace=True)

    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)

    # ----- B: Name Normalization -----
    if 'Name' in combined_df.columns:
        # Last digit = 4-5 → title case
        combined_df['Name'] = combined_df['Name'].astype(str).str.title()

    # ----- E: Handle missing auxiliary data -----
    # Roll number second to last and last digit = 3-4 → drop rows with missing auxiliary info
    auxiliary_cols = ['conversation_status', 'habitat_type', 'diet_type']
    combined_df.dropna(subset=[col for col in auxiliary_cols if col in combined_df.columns], inplace=True)

    return combined_df

# Run the function
if __name__ == "__main__":
    combined_data = Gamma_load_and_integrate()
    print("Data loaded, integrated, normalized, and cleaned successfully!")
    print(combined_data.head())
