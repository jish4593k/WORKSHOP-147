import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


data = loadmat('AdsenseImpressions.mat')

absTop = data['Impr_Abs_Top']
top = data['Impr_Top']
keywords = data['Keyword']


df = pd.DataFrame({'absTop': absTop.flatten(), 'top': top.flatten(), 'keywords': keywords.flatten()})


df = df.dropna()


df['absTop'] = df['absTop'] / 100
df['top'] = df['top'] / 100

X = df[['absTop', 'top']].to_numpy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


num_clusters = 3


kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_scaled)
labels = kmeans.labels_


df['PredictedGroup'] = labels

# Visualize predicted labels
plt.scatter(x=df.absTop, y=df.top, c=df.PredictedGroup)
plt.xlabel("Absolute Top")
plt.ylabel("Top")
plt.show()

# Write data to CSV file (insert the path of your choosing)
df.to_csv("predicted_data.csv", index=False)

# Display cluster labels and the original data
print("Cluster Labels:")
print(labels)
print("\nOriginal Data:")
print(df.head())

# Build a Tkinter window for visualization
window = tk.Tk()
window.title("AdSense Impressions Clustering")
window.geometry("800x600")

# Function to choose a file using file dialog
def choose_file():
    file_path = filedialog.askopenfilename()
    print(f'Selected file: {file_path}')

# Create a button to choose a file
file_button = tk.Button(window, text="Choose File", command=choose_file)
file_button.pack()

window.mainloop()
