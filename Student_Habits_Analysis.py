
"""
Student Habits Analysis
Author: Nicole Mei
Description: Exploratory data analysis, clustering, and classification modeling
to examine how student habits relate to academic performance.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, confusion_matrix



plt.rcParams['figure.dpi'] = 100



# =========================
# Load and Prepare Data
# =========================
df = pd.read_csv("student_habits_performance.csv", na_values=[], keep_default_na=False)
if 'student_id' in df.columns:
    df.drop(columns='student_id', inplace=True)


print("Training data shape:", df.shape)

rows, columns = df.shape
print(f"The data has {columns} attributes and {rows} rows.")



df.head()
df.info()

print("\n" + "-"*50 + "\n")
print(df.isnull().sum())


# =========================
# Exploratory Data Analysis
# =========================

bins = [0, 1, 2, 4, 6, 8]
labels = ["<1", "1–2", "2–4", "4–6", "6–8"]

df['study_bin'] = pd.cut(df['study_hours_per_day'], bins=bins, labels=labels, right=False)

study_score = df.groupby('study_bin', observed=False)['exam_score'].mean()

study_score = study_score.reindex(labels)

print("Average Exam Score by Study Hours:\n")
print(study_score)

ax = study_score.plot(kind='bar', figsize=(10, 6), legend=False, color='blue')
ax.set_title('Average Exam Score by Study Hours (Binned)')
ax.set_ylabel('Average Exam Score')
ax.set_xlabel('Study Hours per Day')
ax.set_xticklabels(labels, rotation=0)
ax.grid(True)
#passing score line of 70
plt.axhline(y=70, color='red', linestyle='--', linewidth=2, label='Passing Score (70)')

plt.legend()

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

df['sleep_bin'] = pd.cut(df['sleep_hours'],
                         bins=[0, 5, 6.5, 8, 10],
                         labels=["<5", "5–6.5", "6.5–8", "8–10"])
sleep_score = df.groupby('sleep_bin', observed=False)['exam_score'].mean()
ordered_sleep_bins = ["<5", "5–6.5", "6.5–8", "8–10"]
sleep_score = sleep_score.reindex(ordered_sleep_bins)


df['exercise_bin'] = pd.cut(df['exercise_frequency'],
                            bins=[-1, 1, 3, 5, 7],
                            labels=["Rarely (0–1)", "Sometimes (2–3)", "Often (4–5)", "Very Often (6–7)"])
exercise_score = df.groupby('exercise_bin', observed=False)['exam_score'].mean()
ordered_ex_bins = ["Rarely (0–1)", "Sometimes (2–3)", "Often (4–5)", "Very Often (6–7)"]
exercise_score = exercise_score.reindex(ordered_ex_bins)

print("Average Exam Score by Sleep Hours:\n")
print(sleep_score.to_string(float_format="{:.2f}".format))

print("\n" + "-"*50 + "\n")

print("Average Exam Score by Exercise Frequency:\n")
print(exercise_score.to_string(float_format="{:.2f}".format))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sleep plot
sleep_score.plot(kind='bar', ax=axes[0], color='mediumseagreen')
axes[0].set_title("Exam Score by Sleep Hours")
axes[0].set_xlabel("Sleep Hours")
axes[0].set_ylabel("Avg Exam Score")
axes[0].set_xticklabels(ordered_sleep_bins, rotation=0)
axes[0].grid(True)

# Exercise plot
exercise_score.plot(kind='bar', ax=axes[1], color='cornflowerblue')
axes[1].set_title("Exam Score by Exercise Frequency")
axes[1].set_xlabel("Exercise Frequency")
axes[1].set_ylabel("Avg Exam Score")
axes[1].set_xticklabels(ordered_ex_bins, rotation=0)
axes[1].grid(True)

plt.tight_layout()
plt.show()


#gender
gender_means= df.groupby('gender')['exam_score'].mean()
print("Average Exam Score by Gender:\n")
print(gender_means)

df.groupby('gender')['exam_score'].mean().plot(kind='bar', title='Average Exam Score by gender')
plt.ylabel('Average Exam Score')
plt.xlabel('gender')
plt.grid(True)
plt.tight_layout()
plt.show()

# age 
df['age_bin'] = pd.cut(df['age'], bins=[16, 18, 20, 22, 25], labels=['17–18', '19–20', '21–22', '23–24'], right=False)

age_means = df.groupby('age_bin', observed=True)['exam_score'].mean()


print("Average Exam Score by Age Group:\n")
print(age_means)

df.groupby('age_bin', observed=True)['exam_score'].mean().plot(kind='bar', title='Average Exam Score by Age Group')
plt.ylabel('Average Exam Score')
plt.xlabel('Age Group')
plt.grid(True)
plt.tight_layout()
plt.show()



bins_social = [0, 2, 4, 8]
labels_social = ['Low (0–2)', 'Medium (2–4)', 'High (4+)']
df['social_media_bin'] = pd.cut(df['social_media_hours'], bins=bins_social, labels=labels_social, right=False)

bins_netflix = [0, 2, 4, 8]
labels_netflix = ['Low (0–2)', 'Medium (2–4)', 'High (4+)']
df['netflix_bin'] = pd.cut(df['netflix_hours'], bins=bins_netflix, labels=labels_netflix, right=False)

print("Average Exam Score by Social Media Usage:")
print(df.groupby('social_media_bin', observed=True)['exam_score'].mean())

print("\nAverage Exam Score by Netflix Usage:")
print(df.groupby('netflix_bin', observed=True)['exam_score'].mean())

social_scores = df.groupby('social_media_bin', observed=True)['exam_score'].mean()
netflix_scores = df.groupby('netflix_bin', observed=True)['exam_score'].mean()

df_combined = pd.DataFrame({
    'Social Media Usage': social_scores,
    'Netflix Usage': netflix_scores
}).T

df_combined = df_combined.T

ax = df_combined.plot(kind='bar', figsize=(10, 6), rot=0)
ax.set_title('Average Exam Score by Usage Type and Level')
ax.set_xlabel('Usage Type')
ax.set_ylabel('Average Exam Score')
ax.legend(title='Usage Level')
ax.grid(axis='y')
plt.tight_layout()
plt.show()




df['pass_fail'] = df['exam_score'].apply(lambda x: 'Pass' if x >= 70 else 'Fail')
sns.countplot(x='pass_fail', data=df, color='blue', order=['Fail', 'Pass'])
plt.title('Exam Performance Distribution')
plt.xlabel('Exam Outcome')
plt.ylabel('Count')


sns.violinplot(x='social_media_hours', y='pass_fail', data=df, hue= 'pass_fail', palette='Paired', legend= False)
plt.title('Social Media Usage by Exam Outcome')
plt.xlabel('Hours of Social Media Usage')
plt.ylabel('Exam Outcome')
plt.grid(True)
plt.tight_layout()
plt.show()


sns.violinplot(x='study_hours_per_day', y='pass_fail', data=df, hue='pass_fail', palette='Paired',legend=False)
plt.title('Study Hours by Exam Outcome')
plt.xlabel('Study Hours per Day')
plt.ylabel('Exam Outcome')
plt.grid(True)
plt.tight_layout()
plt.show()




numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(12, 8)) 

corr = numeric_df.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# =========================
# Clustering
# =========================

X = df.select_dtypes(include='number')

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_pca_3d)
centers = kmeans.cluster_centers_

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=clusters, cmap='Set2', s=20, alpha=0.6)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=80, marker='x', label='Cluster Centers')
ax.set_title("K-Means Clustering in 3D PCA Space")
ax.set_xlabel("Component 0")
ax.set_ylabel("Component 1")
ax.set_zlabel("Component 2")
ax.legend()
plt.tight_layout()
plt.show()

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_pca)
centers = kmeans.cluster_centers_

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
components = [(0, 1), (1, 2), (0, 2)]

for idx, (i, j) in enumerate(components):
    axes[idx].scatter(X_pca[:, i], X_pca[:, j], c=labels, cmap='coolwarm', s=10, alpha=0.6)
    axes[idx].scatter(centers[:, i], centers[:, j], c='black', marker='x', s=100)
    axes[idx].set_xlabel(f'Component {i}')
    axes[idx].set_ylabel(f'Component {j}')

plt.tight_layout()
plt.show()



features_to_plot = ['study_hours_per_day', 'sleep_hours', 'mental_health_rating', 
                    'social_media_hours', 'netflix_hours', 'exercise_frequency']

X = df[features_to_plot]  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


df_clustered = df.copy()
df_clustered['cluster'] = clusters

df_clustered.to_csv("final_kmeans_clusters.csv", index=False)

cluster_stats = df_clustered.groupby('cluster')[features_to_plot + ['exam_score']].mean()
print("Cluster Summary Statistics:\n")
print(cluster_stats)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(features_to_plot):
    sns.histplot(
        data=df_clustered,
        x=feature,
        hue='cluster',
        ax=axes[idx],
        bins=20,
        kde=True,
        palette='Set2',
        multiple='dodge',
        stat='density',
        common_norm=False
    )
    axes[idx].set_title(f'{feature} by Cluster')

    handles, labels = axes[idx].get_legend_handles_labels()
    if handles:
        axes[idx].legend(handles=handles, labels=labels, title='Cluster')

plt.suptitle("K-Means Cluster Statistics", fontsize=16)
plt.tight_layout()
plt.show()



X = df.drop(columns=['exam_score'], errors='ignore').select_dtypes(include='number')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method='ward') 

plt.figure(figsize=(12, 6))
dendrogram(
    linked,
    truncate_mode='lastp',  
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)
plt.title('Hierarchical Clustering Dendrogram (Truncated)')
plt.xlabel('Cluster Size (last 30 merges)')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# =========================
# Classification Models
# =========================


from sklearn.model_selection import train_test_split

X = df[[
    'study_hours_per_day', 'sleep_hours', 'social_media_hours', 'netflix_hours',
    'exercise_frequency', 'mental_health_rating'
]]
y = df['pass_fail']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Support Vector Machine 
svc_model = SVC(kernel='linear', probability=True, random_state=42)

svc_model.fit(X_train, y_train)

y_pred_svc = svc_model.predict(X_test)

accuracy_svc = accuracy_score(y_test, y_pred_svc) * 100

print("=== Support Vector Classifier (SVC) ===")
print(f"Accuracy: {accuracy_svc:.2f}%\n")
print("\nClassification Report:\n", classification_report(y_test, y_pred_svc))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100

print("=== Random Forest Classifier ===")
print(f"Accuracy: {accuracy_rf:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

importances = rf_model.feature_importances_

feature_names = X_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, color='steelblue')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()



# K-Nearest Neighbors (KNN)
best_k = 0
best_accuracy = 0

print("=== Tuning K for KNN ===\n")
for k in [3, 5, 7, 10]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"k = {k}: Accuracy = {acc:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print(f"\nBest k value: {best_k} with Accuracy: {best_accuracy:.2f}%\n")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100

print("=== Final K-Nearest Neighbors (KNN) ===")
print(f"Accuracy: {accuracy_knn:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))



# Gaussian Naive Bayes
gnb_model = GaussianNB()

gnb_model.fit(X_train, y_train)

y_pred_gnb = gnb_model.predict(X_test)

accuracy_gnb = accuracy_score(y_test, y_pred_gnb) * 100

print("=== Gaussian Naive Bayes (GNB) ===")
print(f"Accuracy: {accuracy_gnb:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred_gnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gnb))


# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)

accuracy_gb = accuracy_score(y_test, y_pred_gb) * 100

print("=== Gradient Boosting Classifier ===")
print(f"Accuracy: {accuracy_gb:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))



# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)

accuracy_ada = accuracy_score(y_test, y_pred_ada) * 100
print("=== AdaBoost Classifier ===")
print(f"Accuracy: {accuracy_ada:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred_ada))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ada))

importances = ada_model.feature_importances_
feature_names = X_train.columns  # Make sure X_train is a DataFrame

feat_imp = pd.Series(importances, index=feature_names).sort_values()

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='steelblue')
plt.title("AdaBoost Feature Importances (Full Features)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# Voting Classifier
svc = SVC(probability=True, kernel='linear', random_state=42)
gnb = GaussianNB()
abc = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('svc', svc), ('gnb', gnb), ('abc', abc), ('rf', rf)],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

y_pred_vote = voting_clf.predict(X_test)
accuracy_vote = accuracy_score(y_test, y_pred_vote) * 100

print("=== Voting Classifier ===")
print(f"Accuracy: {accuracy_vote:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred_vote))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_vote))


# PCA visualization for voting classifier

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # y should be 'Pass'/'Fail' or similar

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_encoded, test_size=0.2, random_state=42
)

svc = SVC(probability=True, kernel='linear', random_state=42)
gnb = GaussianNB()  # No random_state needed
abc = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

voting_clf_soft = VotingClassifier(
    estimators=[('svc', svc), ('gnb', gnb), ('abc', abc), ('rf', rf)],
    voting='soft'
)

voting_clf_hard = VotingClassifier(
    estimators=[('svc', svc), ('gnb', gnb), ('abc', abc), ('rf', rf)],
    voting='hard'
)

voting_clf_soft.fit(X_train_pca, y_train_pca)
voting_clf_hard.fit(X_train_pca, y_train_pca)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))
grid = np.c_[xx.ravel(), yy.ravel()]

Z_soft = voting_clf_soft.predict(grid).reshape(xx.shape)
Z_hard = voting_clf_hard.predict(grid).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].contourf(xx, yy, Z_soft, alpha=0.3, cmap=ListedColormap(['#87CEFA', '#FFA07A', '#98FB98']))
axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, edgecolor='k', cmap=ListedColormap(['#0000FF', '#FF4500', '#008000']))
axes[0].set_title("Soft Voting")
axes[0].set_xlabel("PCA Feature 1")
axes[0].set_ylabel("PCA Feature 2")

axes[1].contourf(xx, yy, Z_hard, alpha=0.3, cmap=ListedColormap(['#87CEFA', '#FFA07A', '#98FB98']))
axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, edgecolor='k', cmap=ListedColormap(['#0000FF', '#FF4500', '#008000']))
axes[1].set_title("Hard Voting")
axes[1].set_xlabel("PCA Feature 1")
axes[1].set_ylabel("PCA Feature 2")

plt.tight_layout()
plt.show()

print(y.unique())

# =========================
# Model Comparison
# =========================

models = pd.DataFrame({
    'Model': [
        'Support Vector Machine (SVC)','K-Nearest Neighbors (KNN)','Random Forest','Gaussian Naive Bayes','Gradient Boosting','AdaBoost','Voting Classifier'],
    'Score': [
        accuracy_svc, accuracy_knn, accuracy_rf, accuracy_gnb, accuracy_gb, accuracy_ada, accuracy_vote]
})

models = models.sort_values(by='Score', ascending=False).reset_index(drop=True)
print(models)