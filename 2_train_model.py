import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train():
    print("==================================================")
    print("       HYBRID MERGER & RANDOM FOREST TRAINER")
    print("==================================================")
    
    # Load Kaggle & Custom Datasets
    kaggle_path = 'dataset/kaggle_features.csv'
    custom_path = 'dataset/custom_features.csv'
    
    df_kaggle = pd.read_csv(kaggle_path) if os.path.exists(kaggle_path) else pd.DataFrame()
    df_custom = pd.read_csv(custom_path) if os.path.exists(custom_path) else pd.DataFrame()

    if df_kaggle.empty and df_custom.empty:
        print("[Error] No dataset found! Please collect data or provide features CSV.")
        return

    df_master = pd.concat([df_kaggle, df_custom], ignore_index=True)
    print(f'[Data Info] Total merged samples: {len(df_master)}')
    print(f'            - From Kaggle dataset: {len(df_kaggle)}')
    print(f'            - From Custom webcam : {len(df_custom)}')

    X = df_master[['avg_ear', 'l_ear', 'r_ear', 'mor']]
    y = df_master['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[Training] Training RandomForestClassifier (100 Trees)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f'\n[Results] Model Accuracy: {acc * 100:.2f}%\n')
    target_names = ['ACTIVE (0)', 'DROWSY (1)', 'YAWNING (2)']
    print(classification_report(y_test, preds, target_names=target_names))

    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/rfc_drowsiness_model.pkl'
    joblib.dump(model, model_save_path)
    print(f'[Saved] Trained model successfully saved to: {model_save_path}')

if __name__ == '__main__':
    train()
