from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Veri Setini Yükleme
data_path = os.path.join(os.path.dirname(__file__), 'data.xlsx')
df = pd.read_excel(data_path)

def exclude_columns(df, exclude_keywords):
    exclude_cols = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in exclude_keywords)]
    df_filtered = df.drop(columns=exclude_cols)
    return df_filtered, exclude_cols

def concat_disqualified_col(df_processed, df_original, exclude_cols):
    df_final = pd.concat([df_processed, df_original[exclude_cols]], axis=1)
    return df_final

# Eksik Veri Görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('Eksik Veriler')
plt.show()

# Veri temizleme & doldurma
df , _ = exclude_columns(df,["id","zaman damgasi","zaman damgası","Yaşadığınız yer (Mahayıralle/Sokak)"])

for col in df.columns:
    if df[col].dtype == 'object':  # Kategorik değişkenler
        df[col].fillna(df[col].mode()[0], inplace=True)  # En sık tekrar eden değerle doldur
    else:  # Sayısal değişkenler
        if df[col].isnull().sum() / len(df) > 0.3:  # %30'dan fazla eksik varsa sütunu sil
            df.drop(columns=[col], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Medyan ile doldur

# Aykırı Değerleri Tespit Etme (IQR & Z-Score)
plt.figure(figsize=(10, 5))
sns.boxplot(data=df.select_dtypes(include=[np.number]))
plt.xticks(rotation=90)
plt.title("Aykırı Değerler")
plt.show()

# Aykırı değerleri temizleme (Winsorizing - uç değerleri kırpma)
for col in df.select_dtypes(include=[np.number]).columns:
    z_scores = np.abs(stats.zscore(df[col]))  # Z-score hesaplama
    df[col] = np.where(z_scores > 3, df[col].median(), df[col])  # Z-score > 3 olanları medyan ile değiştir

df_processed = df
df_processed_final, disqualified_columns = exclude_columns(df_processed, ["yaş"])

# Kategorik Değişkenleri Belirleme
categorical_cols = df_processed_final.select_dtypes(include=['object']).columns
binary_cols = [col for col in categorical_cols if df_processed_final[col].nunique() == 2]
multi_class_cols = [col for col in categorical_cols if df_processed_final[col].nunique() > 2]
print("-------------------------------------")
print(categorical_cols)
print("-------------------------------------")
print(binary_cols)
print("-------------------------------------")
print(multi_class_cols)
# Binary Kategorik Değişkenler için Label Encoding
label_encoder = LabelEncoder()
for col in binary_cols:
    df_processed_final.loc[:, col] = label_encoder.fit_transform(df_processed_final[col])

# Çok Kategorili Değişkenler için One-Hot Encoding
df_processed_final = pd.get_dummies(df_processed_final, columns=multi_class_cols, drop_first=True)

# Güncellenmiş Kategorik Değişken Listesi
categorical_cols = list(binary_cols) + list(multi_class_cols)

# Sayısal Değişkenleri Ölçekleme
num_cols = [col for col in df_processed_final.select_dtypes(include=[np.number]).columns if col not in categorical_cols]
if len(num_cols) > 0:
    scaler = StandardScaler()
    df_processed_final[num_cols] = scaler.fit_transform(df_processed_final[num_cols])
else:
    print("Sayısal değişken bulunamadı, StandardScaler uygulanmadı.")

df_final = concat_disqualified_col(df_processed_final, df_processed, disqualified_columns)
# Temizlenmiş Veri Setini Kaydetme
df_final.to_excel(os.path.join(os.path.dirname(__file__), 'processed_data.xlsx'), index=False)

print("\nÖn İşlenmiş Veri Seti:\n", df_final.head())