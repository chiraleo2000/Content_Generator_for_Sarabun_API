import pandas as pd

def read_and_clean_excel(file_path):
    # Read the first sheet of the Excel file
    df = pd.read_excel(file_path)

    # Remove rows with duplicate words in the 'ภาษาราชการ' column
    df_cleaned = df.drop_duplicates(subset=['ภาษาราชการ'], keep='first')

    # Display the cleaned DataFrame
    print(df_cleaned)

    # Optional: Save the cleaned DataFrame to a new Excel file
    df_cleaned.to_excel('List_of_words_to_changes.xlsx', index=False)

    return df_cleaned

if __name__ == '__main__':
    read_and_clean_excel('List_of_words_to_changes.xlsx')
