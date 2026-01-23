import pandas as pd
import os

# Read the CSV file
csv_file = "SUGAR UPTO 1000MG_approx.csv"
df = pd.read_csv(csv_file)

print("CSV File Structure:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Get frequency column
freq_col = df.iloc[:, 0].tolist()

# Get all mg level columns (skip the first frequency column)
mg_columns = df.columns[1:]

print(f"Creating Excel files for {len(mg_columns)} mg levels...")
print("=" * 60)

# Create an Excel file for each mg level
for mg_col in mg_columns:
    try:
        # Get the mg level from column name (e.g., "100MG" -> "100")
        mg_value = mg_col.replace("MG", "")
        
        # Create a dataframe with Frequency and Return Loss data
        data = {
            "Frequency [GHz]": freq_col,
            "Return Loss [dB]": df[mg_col].tolist()
        }
        
        new_df = pd.DataFrame(data)
        
        # Create Excel filename
        excel_filename = f"glucose_{mg_value}mg.xlsx"
        
        # Save to Excel
        new_df.to_excel(excel_filename, index=False, sheet_name=f"{mg_value}MG")
        
        print(f"✓ Created: {excel_filename} ({len(new_df)} rows)")
        
    except Exception as e:
        print(f"✗ Error creating {excel_filename}: {str(e)}")

print("=" * 60)
print(f"All {len(mg_columns)} Excel files created successfully!")
print("\nFiles created:")
for mg_col in mg_columns:
    mg_value = mg_col.replace("MG", "")
    print(f"  • glucose_{mg_value}mg.xlsx")
