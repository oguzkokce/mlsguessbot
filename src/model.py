"""
MLS Maç Tahmin Sistemi - Model Eğitimi Modülü
Bu modül makine öğrenmesi modellerini eğitir ve değerlendirir.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class MLSModelTrainer:
    """MLS maç tahmin modellerini eğiten sınıf"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Args:
            data_dir: Veri klasörü yolu
            models_dir: Model klasörü yolu
        """
        self.data_dir = data_dir
        self.processed_dir = f"{data_dir}/processed"
        self.models_dir = models_dir
        
        # Model klasörünü oluştur
        os.makedirs(models_dir, exist_ok=True)
        
        # Model sonuçları
        self.models = {}
        self.results = {}
        
        logger.info("MLS Model Trainer başlatıldı")
    
    def load_processed_data(self) -> pd.DataFrame:
        """İşlenmiş veriyi yükle"""
        try:
            df = pd.read_csv(f"{self.processed_dir}/processed_matches.csv")
            logger.info(f"İşlenmiş veri yüklendi: {len(df)} satır, {len(df.columns)} sütun")
            return df
        except FileNotFoundError:
            logger.error("İşlenmiş veri bulunamadı!")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Özellikleri ve hedef değişkeni hazırla
        
        Args:
            df: İşlenmiş veri DataFrame'i
            
        Returns:
            X: Özellik matrisi, y: Hedef değişken
        """
        # Hedef değişkeni hariç tut
        exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'result', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        logger.info(f"Özellik matrisi boyutu: {X.shape}")
        logger.info(f"Hedef değişken dağılımı: {np.bincount(y)}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Veriyi eğitim ve test setlerine böl
        
        Args:
            X: Özellik matrisi
            y: Hedef değişken
            test_size: Test seti oranı
            random_state: Rastgele tohum
            
        Returns:
            Eğitim ve test setleri
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Eğitim seti boyutu: {X_train.shape}")
        logger.info(f"Test seti boyutu: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Random Forest modelini eğit
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedef değişkeni
            
        Returns:
            Eğitilmiş Random Forest modeli
        """
        logger.info("Random Forest modeli eğitiliyor...")
        
        # Hiperparametre arama alanı
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        }
        
        # Grid Search
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        logger.info(f"En iyi parametreler: {grid_search.best_params_}")
        logger.info(f"En iyi CV skoru: {grid_search.best_score_:.4f}")
        
        return best_rf
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """
        XGBoost modelini eğit
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedef değişkeni
            
        Returns:
            Eğitilmiş XGBoost modeli
        """
        logger.info("XGBoost modeli eğitiliyor...")
        
        # Hiperparametre arama alanı
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'random_state': [42]
        }
        
        # Grid Search
        xgb_model = XGBClassifier()
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        logger.info(f"En iyi parametreler: {grid_search.best_params_}")
        logger.info(f"En iyi CV skoru: {grid_search.best_score_:.4f}")
        
        return best_xgb
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Logistic Regression modelini eğit
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedef değişkeni
            
        Returns:
            Eğitilmiş Logistic Regression modeli
        """
        logger.info("Logistic Regression modeli eğitiliyor...")
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Hiperparametre arama alanı
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'random_state': [42]
        }
        
        # Grid Search
        lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        best_lr = grid_search.best_estimator_
        logger.info(f"En iyi parametreler: {grid_search.best_params_}")
        logger.info(f"En iyi CV skoru: {grid_search.best_score_:.4f}")
        
        # Scaler'ı model ile birlikte sakla
        best_lr.scaler = scaler
        
        return best_lr
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict:
        """
        Modeli değerlendir
        
        Args:
            model: Eğitilmiş model
            X_test: Test özellikleri
            y_test: Test hedef değişkeni
            model_name: Model adı
            
        Returns:
            Değerlendirme sonuçları
        """
        logger.info(f"{model_name} modeli değerlendiriliyor...")
        
        # Tahmin yap
        if hasattr(model, 'scaler'):
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Metrikleri hesapla
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Sınıf bazında F1 skorları
        f1_scores = f1_score(y_test, y_pred, average=None)
        
        # Sonuçları sakla
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_home': f1_scores[0],
            'f1_draw': f1_scores[1],
            'f1_away': f1_scores[2],
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_values': y_test
        }
        
        logger.info(f"{model_name} Sonuçları:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Macro: {f1_macro:.4f}")
        logger.info(f"  F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"  F1 Home: {f1_scores[0]:.4f}")
        logger.info(f"  F1 Draw: {f1_scores[1]:.4f}")
        logger.info(f"  F1 Away: {f1_scores[2]:.4f}")
        
        return results
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """
        Confusion matrix oluştur ve kaydet
        
        Args:
            y_true: Gerçek değerler
            y_pred: Tahmin edilen değerler
            model_name: Model adı
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Görselleştir
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Home', 'Draw', 'Away'],
                   yticklabels=['Home', 'Draw', 'Away'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Gerçek Değerler')
        plt.xlabel('Tahmin Edilen Değerler')
        
        # Kaydet
        plt.savefig(f"{self.models_dir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix kaydedildi: {model_name}_confusion_matrix.png")
    
    def get_feature_importance(self, model, feature_names: List[str], model_name: str) -> pd.DataFrame:
        """
        Özellik önemini al
        
        Args:
            model: Eğitilmiş model
            feature_names: Özellik isimleri
            model_name: Model adı
            
        Returns:
            Özellik önem DataFrame'i
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            logger.warning(f"{model_name} için özellik önem bilgisi alınamadı")
            return pd.DataFrame()
        
        # DataFrame oluştur
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Kaydet
        importance_df.to_csv(f"{self.models_dir}/{model_name}_feature_importance.csv", index=False)
        
        logger.info(f"Özellik önem sıralaması kaydedildi: {model_name}_feature_importance.csv")
        
        return importance_df
    
    def save_model(self, model, model_name: str):
        """
        Modeli kaydet
        
        Args:
            model: Eğitilmiş model
            model_name: Model adı
        """
        model_path = f"{self.models_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model kaydedildi: {model_path}")
    
    def train_all_models(self) -> Dict:
        """
        Tüm modelleri eğit
        
        Returns:
            Eğitilmiş modeller ve sonuçları
        """
        logger.info("Tüm modeller eğitiliyor...")
        
        # Veriyi yükle
        df = self.load_processed_data()
        if df.empty:
            logger.error("Veri yüklenemedi!")
            return {}
        
        # Özellikleri hazırla
        X, y = self.prepare_features(df)
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Özellik isimleri
        exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'result', 'target']
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        # Modelleri eğit
        models_to_train = {
            'RandomForest': self.train_random_forest,
            'XGBoost': self.train_xgboost,
            'LogisticRegression': self.train_logistic_regression
        }
        
        for model_name, train_func in models_to_train.items():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"{model_name} eğitimi başlıyor...")
                
                # Modeli eğit
                model = train_func(X_train, y_train)
                
                # Modeli değerlendir
                results = self.evaluate_model(model, X_test, y_test, model_name)
                
                # Confusion matrix oluştur
                self.create_confusion_matrix(y_test, results['predictions'], model_name)
                
                # Özellik önemini al
                importance_df = self.get_feature_importance(model, feature_names, model_name)
                
                # Modeli kaydet
                self.save_model(model, model_name)
                
                # Sonuçları sakla
                self.models[model_name] = model
                self.results[model_name] = results
                
                logger.info(f"{model_name} eğitimi tamamlandı!")
                
            except Exception as e:
                logger.error(f"{model_name} eğitimi sırasında hata: {e}")
        
        return self.models
    
    def compare_models(self) -> pd.DataFrame:
        """
        Modelleri karşılaştır
        
        Returns:
            Karşılaştırma DataFrame'i
        """
        if not self.results:
            logger.warning("Karşılaştırılacak sonuç bulunamadı!")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1 Macro': results['f1_macro'],
                'F1 Weighted': results['f1_weighted'],
                'F1 Home': results['f1_home'],
                'F1 Draw': results['f1_draw'],
                'F1 Away': results['f1_away']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Macro', ascending=False)
        
        # Kaydet
        comparison_df.to_csv(f"{self.models_dir}/model_comparison.csv", index=False)
        
        logger.info("Model karşılaştırması kaydedildi: model_comparison.csv")
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[str, object]:
        """
        En iyi modeli döndür
        
        Returns:
            En iyi model adı ve modeli
        """
        if not self.results:
            logger.warning("Sonuç bulunamadı!")
            return None, None
        
        # F1 Macro skoruna göre en iyi modeli bul
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1_macro'])
        best_model = self.models[best_model_name]
        
        logger.info(f"En iyi model: {best_model_name}")
        logger.info(f"F1 Macro skoru: {self.results[best_model_name]['f1_macro']:.4f}")
        
        return best_model_name, best_model
    
    def predict_match(self, features: np.ndarray, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maç tahmini yap
        
        Args:
            features: Özellik vektörü
            model_name: Model adı (None ise en iyi model kullanılır)
            
        Returns:
            Tahmin ve olasılıklar
        """
        if model_name is None:
            model_name, model = self.get_best_model()
        else:
            model = self.models.get(model_name)
        
        if model is None:
            logger.error("Model bulunamadı!")
            return None, None
        
        # Tahmin yap
        if hasattr(model, 'scaler'):
            features_scaled = model.scaler.transform(features)
            prediction = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
        else:
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)
        
        return prediction, probabilities

def main():
    """Ana fonksiyon - model eğitimi işlemini başlatır"""
    trainer = MLSModelTrainer()
    
    # Tüm modelleri eğit
    models = trainer.train_all_models()
    
    if models:
        # Modelleri karşılaştır
        comparison_df = trainer.compare_models()
        
        print("\n" + "="*50)
        print("MODEL KARŞILAŞTIRMASI")
        print("="*50)
        print(comparison_df.to_string(index=False))
        
        # En iyi modeli göster
        best_name, best_model = trainer.get_best_model()
        print(f"\nEn iyi model: {best_name}")
        
        print("\n" + "="*50)
        print("Model eğitimi tamamlandı! 🎯")
    else:
        print("Model eğitimi başarısız!")

if __name__ == "__main__":
    main() 