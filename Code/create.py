import pandas as pd

# Load the CSV file
input_file = 'creditcard.csv'  # Replace with your actual file name
df = pd.read_csv(input_file)

# Take a random sample of 10,000 records
sample_df = df.sample(n=10000, random_state=42)

# Save to a new CSV file
output_file = 'sample_10000_records.csv'
sample_df.to_csv(output_file, index=False)

print(f"Saved 10,000 sampled records to {output_file}")
