# ===============================
# GLOBAL TERRORISM DATA ANALYSIS PROJECT
# ===============================

# ---- Import Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---- Step 1: Load Dataset ----
path = r"C:/Users/hp/OneDrive/Documents/globalterrorismdb_0718dist.csv"
df = pd.read_csv(path, encoding='latin1', low_memory=False)

print("Dataset Loaded Successfully ✅")
print("Shape of dataset:", df.shape)
print(df.head())

# ---- Step 2: Data Cleaning ----
df = df[['iyear', 'imonth', 'iday', 'country_txt', 'region_txt',
         'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt',
         'nkill', 'nwound', 'latitude', 'longitude']]

df.rename(columns={
    'iyear': 'Year',
    'imonth': 'Month',
    'iday': 'Day',
    'country_txt': 'Country',
    'region_txt': 'Region',
    'attacktype1_txt': 'AttackType',
    'targtype1_txt': 'TargetType',
    'weaptype1_txt': 'WeaponType',
    'nkill': 'Killed',
    'nwound': 'Wounded'
}, inplace=True)

df['Killed'].fillna(0, inplace=True)
df['Wounded'].fillna(0, inplace=True)
df['Casualties'] = df['Killed'] + df['Wounded']
df.dropna(subset=['latitude', 'longitude'], inplace=True)

print("\nData cleaned successfully ✅")

# ---- Step 3: Exploratory Data Analysis ----
plt.figure(figsize=(12, 6))
sns.countplot(x='Region', data=df, order=df['Region'].value_counts().index, palette='coolwarm')
plt.xticks(rotation=75)
plt.title("Number of Terrorist Attacks by Region")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
df.groupby('Year').size().plot(kind='line', marker='o', color='darkred')
plt.title("Global Terrorism Trends Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Attacks")
plt.grid(True)
plt.show()

top_countries = df['Country'].value_counts().nlargest(10)
top_countries.plot(kind='bar', figsize=(10, 5), color='teal')
plt.title("Top 10 Countries Affected by Terrorism")
plt.xlabel("Country")
plt.ylabel("Number of Attacks")
plt.tight_layout()
plt.show()

# ---- Step 4: Geospatial Visualization ----
print("\nCreating Geospatial Map (may take a few seconds)...")
world_map = folium.Map(location=[20, 0], zoom_start=2, tiles='Cartodb dark_matter')

for _, row in df.sample(1000).iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='red',
        fill=True,
        fill_color='red',
        popup=f"{row['Country']} | {row['AttackType']}"
    ).add_to(world_map)

world_map.save("terrorism_hotspots_map.html")
print("✅ Map saved as 'terrorism_hotspots_map.html' (open in browser)")

# ---- Step 5: Statistical Correlation ----
corr = df[['Killed', 'Wounded', 'Casualties']].corr()
sns.heatmap(corr, annot=True, cmap='magma')
plt.title("Correlation Heatmap")
plt.show()

# ---- Step 6: Clustering (K-Means & DBSCAN) ----
X = df[['Killed', 'Wounded', 'Casualties']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("K-Means Silhouette Score:", silhouette_score(X_scaled, df['Cluster']))

sns.scatterplot(x='Killed', y='Wounded', hue='Cluster', data=df, palette='Set2')
plt.title("K-Means Clustering of Terrorism Impact")
plt.show()

# ---- Step 7: Association Rule Mining ----
df_assoc = pd.get_dummies(df[['AttackType', 'WeaponType', 'TargetType']])
frequent_itemsets = fpgrowth(df_assoc, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.5)
print("\nTop Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# ---- Step 8: Predictive Modeling (Bonus) ----
df['HighCasualty'] = (df['Casualties'] >= 10).astype(int)
X = pd.get_dummies(df[['Region', 'AttackType', 'WeaponType']], drop_first=True)
y = df['HighCasualty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n✅ PROJECT EXECUTED SUCCESSFULLY ✅")
print("📊 Visuals Displayed | 🌍 Map Saved | 🔗 Rules Mined | 🤖 Model Evaluated")
