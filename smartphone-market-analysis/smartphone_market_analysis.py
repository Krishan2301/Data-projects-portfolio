# ================================
# 📚 IMPORTING LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 📂 LOAD DATA
# ================================
df = pd.read_csv('/content/smartphones - smartphones.csv')
print(df.head(3))

# ================================
# 🧹 INITIAL CLEANING & FEATURE ENGINEERING
# ================================

# Extract company name from model name
df['company'] = df['model'].str.split().str[0]

# Extract RAM and ROM separately from 'ram' column
temp_ram = df['ram'].str.strip().str.split().str[0:-1]
df['RAM'] = temp_ram.str[0] + ' ' + temp_ram.str[1]
df['ROM'] = temp_ram.str[-2] + ' ' + temp_ram.str[-1]

# Clean and convert price from string to integer
df['cor_price'] = df['price'].str[1:].str.replace(',', '').astype(int)

# ================================
# ⚠️ HANDLE MISCATEGORIZED VALUES IN TEXT FIELDS
# ================================

# Example fix for incorrectly shifted 'os' value
temp = df[df['os'] == 'No FM Radio']
temp_os = temp['card']
temp['os'] = temp_os
temp['card'] = 'Memory Card Not Supported'
df.loc[temp.index] = temp

# ================================
# 🧠 CATEGORIZE OS COLUMN
# ================================

def categorize_os(value):
    val = str(value).lower()
    if 'android' in val:
        return 'android'
    elif 'ios' in val:
        return 'ios'
    elif any(keyword in val for keyword in ['browser', 'camera', 'memory', 'bluetooth']):
        return 'trash'
    else:
        return 'other'

df['os_group'] = df['os'].apply(categorize_os)

# Remove rows with invalid OS values
df = df[df['os_group'] != 'trash'].reset_index(drop=True)

# ================================
# 🧩 HANDLE MISSING VALUES
# ================================

# Fill missing ratings with median
df['rating'].fillna(df['rating'].median(), inplace=True)

# Drop any remaining nulls
df.dropna(inplace=True)

# ================================
# 🧬 FINAL FEATURE ENGINEERING
# ================================

# Remove rare 'card' entries
card_counts = df['card'].value_counts()
cards_to_remove = card_counts[card_counts < 3].index
df = df[~df['card'].isin(cards_to_remove)]

# Clean 'camera' field
df = df[~df['camera'].str.contains('Memory', na=False)].copy()

# Fix processor data issue
df.loc[774, 'processor'] = 'Samsung ,Quad Core, 1.5 GHz Processor'

# Feature extraction
df['Processor_com'] = df['processor'].str.split().str[0]
df['p_camera'] = df['camera'].str[0:5]
df['battery_cap'] = df['battery'].str.strip().str.split().str[0:2].str.join(' ')
df['charging'] = df['battery'].str.split(' with ').str[-1]
df['display_size'] = df['display'].str.split(' inches').str[0].str.strip()

# Drop old raw columns
df_cleaned = df.drop(columns=['price', 'ram', 'battery', 'display', 'camera', 'os'])

# Feature flags from 'sim'
df_cleaned['has_5g'] = df_cleaned['sim'].str.contains('5G', na=False)
df_cleaned['has_nfc'] = df_cleaned['sim'].str.contains('NFC', na=False)
df_cleaned['has_ir_blaster'] = df_cleaned['sim'].str.contains('IR Blaster', na=False)

# ================================
# 📊 VISUALIZATIONS
# ================================

# Rating Distribution
sns.histplot(data=df_cleaned, x='rating', kde=True)
plt.title('Distribution of Ratings')
plt.show()

# Price vs Rating
sns.scatterplot(data=df_cleaned, x='cor_price', y='rating')
plt.title('Price vs Rating')
plt.xlim(0, 50000)
plt.show()

# Battery Capacity Distribution
df_cleaned['battery_cap_mAh'] = df_cleaned['battery_cap'].str.replace(' mAh', '').astype(float)
df_cleaned['battery_cap_mAh'].value_counts().sort_index().plot(kind='bar')
plt.title('Battery Capacity Distribution')
plt.show()

# Feature Prevalence
for feature in ['has_5g', 'has_nfc', 'has_ir_blaster']:
    sns.countplot(x=df_cleaned[feature])
    plt.title(f'Feature Presence: {feature}')
    plt.show()

# Display Size
df_cleaned['display_size'] = pd.to_numeric(df_cleaned['display_size'], errors='coerce')
sns.histplot(data=df_cleaned, x='display_size', kde=True)
plt.title('Display Size Distribution')
plt.show()

# Top Smartphone Companies
df_cleaned['company'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top Smartphone Brands')
plt.show()

# RAM/ROM Distributions
def convert_to_gb(val):
    if isinstance(val, str):
        val = val.replace(' inbuilt', '')
        if 'TB' in val:
            return float(val.replace(' TB', '')) * 1024
        if 'GB' in val:
            return float(val.replace(' GB', ''))
        if 'MB' in val:
            return float(val.replace(' MB', '')) / 1024
    return np.nan

df_cleaned['RAM_GB'] = df_cleaned['RAM'].apply(convert_to_gb)
df_cleaned['ROM_GB'] = df_cleaned['ROM'].apply(convert_to_gb)

df_cleaned['RAM_GB'].value_counts().sort_index().plot(kind='bar')
plt.title('RAM Distribution')
plt.show()

df_cleaned['ROM_GB'].value_counts().sort_index().plot(kind='bar')
plt.title('ROM Distribution')
plt.show()

# Primary Camera MP Distribution
df_cleaned['primary_camera_MP'] = pd.to_numeric(
    df_cleaned['p_camera'].str.replace(' MP', '').str.replace('MP', ''), errors='coerce')
df_cleaned['primary_camera_MP'].value_counts().sort_index().plot(kind='bar')
plt.title('Primary Camera Megapixels')
plt.show()

# ================================
# 📤 DATA EXPORT & SUMMARY
# ================================

# Export cleaned data
df_cleaned.to_csv('Mobile_Market_Analysis.csv', index=False)
