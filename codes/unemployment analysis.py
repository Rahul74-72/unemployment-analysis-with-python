import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/HP/Desktop/project/archive/Unemployment in India.csv")

df.columns = df.columns.str.strip().str.lower()

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df.dropna(subset=['date'], inplace=True)

df.dropna(inplace=True)

unemployment_trend = df.groupby('date')['estimated unemployment rate (%)'].mean()


print(unemployment_trend.tail())

plt.figure(figsize=(12, 6))
plt.plot(unemployment_trend.index, unemployment_trend.values, marker='o')
plt.title('Average Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.grid(True)
plt.show()

latest_date = df['date'].max()
latest_data = df[df['date'] == latest_date]

plt.figure(figsize=(14, 7))
sns.barplot(data=latest_data.sort_values('estimated unemployment rate (%)', ascending=False),
            x='estimated unemployment rate (%)', y='region', palette='viridis')
plt.title(f'Unemployment Rate by State on {latest_date.date()}')
plt.xlabel('Unemployment Rate')
plt.ylabel('State')
plt.show()

df['Timestamp'] = df['date'].map(pd.Timestamp.timestamp)
X = df[['Timestamp']]

y = df['estimated unemployment rate (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Unemployment Rate")
plt.ylabel("Predicted Unemployment Rate")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()