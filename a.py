import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a CSV file named 'dataset.csv' in the same directory
df = pd.read_csv('dataset.csv')

# Check for empty fields
empty_fields = df.isnull().any().any()
if empty_fields:
    print("There are empty fields in the dataset.")
else:
    print("There are no empty fields in the dataset.")

# Plot every field against the 'Possibility' column directly
for column in df.columns:
    if column != 'Possibility':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Possibility', y=column, data=df)
        plt.title(f'{column} vs Possibility')
        plt.show()

# Assuming you have a DataFrame named 'df' with a column 'Possibility'
df['criminal'] = df['Possibility'].apply(lambda x: 0 if x == '<=0.5' else 1)

# Now the 'criminal' column is added to your existing DataFrame 'df'
df = df.drop('Possibility', axis=1)
print("*******************************")
print(df.iloc[300:351])

for column in df.columns:
    if df[column].dtype in [np.float64, np.int64]:  # Only apply to numeric columns
        average = df[column].mean()
        median = df[column].median()
        mode = df[column].mode()[0]  # mode() returns a Series, take the first value
        print(f"Column: {column}")
        print(f"  Average: {average}")
        print(f"  Median: {median}")
        print(f"  Mode: {mode}")

# Fill missing values with the median for each column
fill_values = {
    'age': 37.0,
    'educationno': 10.0,
    'capitalgain': 0.0,
    'capitalloss': 0.0,
    'hoursperweek': 40.0
}
df.fillna(fill_values, inplace=True)

# Fill missing values in categorical columns with the mode
categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex','native']
mode_values = {column: df[column].mode()[0] for column in categorical_columns}
df.fillna(mode_values, inplace=True)

print("*******************************")
print(df.iloc[300:351])
# Print unique entries for all columns
print("*******************************")
# for column in df.columns:
#     unique_entries = df[column].unique()
#     print(f"Column: {column}")
#     print(f"  Unique entries: {unique_entries}")


# Encode categorical columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print("*******************************")
print(df.iloc[300:351])

# print(df.iloc[300:351])
# Save the preprocessed DataFrame to a new CSV file
df.to_csv('preprocessed_dataset.csv', index=False)

# for column in df.columns:
#     if column != 'criminal':
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='criminal', y=column, data=df)
#         plt.title(f'{column} vs Criminal')
#         plt.show()