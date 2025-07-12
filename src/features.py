"""
MLS Maç Tahmin Sistemi - Özellik Mühendisliği Modülü
Bu modül maç tahminleri için gelişmiş özellikler oluşturur.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from loguru import logger
warnings.filterwarnings('ignore')

class MLSFeatureEngineer:
    """MLS maç tahminleri için özellik mühendisliği sınıfı"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Veri klasörü yolu
        """
        self.data_dir = data_dir
        self.raw_dir = f"{data_dir}/raw"
        self.processed_dir = f"{data_dir}/processed"
        
        logger.info("MLS Feature Engineer başlatıldı")
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Ham verileri yükle"""
        data = {}
        
        # Maç verilerini yükle
        try:
            data['matches'] = pd.read_csv(f"{self.raw_dir}/kaggle_mls_matches.csv")
            data['matches']['date'] = pd.to_datetime(data['matches']['date'])
            logger.info(f"Maç verileri yüklendi: {len(data['matches'])} satır")
        except FileNotFoundError:
            logger.warning("Maç verileri bulunamadı")
            data['matches'] = pd.DataFrame()
        
        # Maaş verilerini yükle
        try:
            data['salaries'] = pd.read_csv(f"{self.raw_dir}/mlspa_salaries.csv")
            logger.info(f"Maaş verileri yüklendi: {len(data['salaries'])} satır")
        except FileNotFoundError:
            logger.warning("Maaş verileri bulunamadı")
            data['salaries'] = pd.DataFrame()
        
        return data
    
    def calculate_team_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Takım form skorlarını hesapla (son N maç)
        """
        logger.info(f"Takım form skorları hesaplanıyor (pencere: {window})")
        
        # Eğer form sütunları varsa kullan, yoksa hesapla
        if 'home_form' in df.columns and 'away_form' in df.columns:
            # Mevcut form verilerini kullan
            df['form_diff'] = df['home_form'] - df['away_form']
        else:
            # Form hesaplama için yardımcı fonksiyon
            def calculate_form(group):
                if len(group) < window:
                    return 0.5  # Varsayılan değer
                return group.tail(window)['result'].map({'Home': 1, 'Away': 0, 'Draw': 0.5}).mean()
            
            # Ev sahibi form
            home_form = df.groupby('home_team').apply(calculate_form).reset_index()
            home_form.columns = ['team', 'home_form_avg']
            
            # Deplasman form
            away_form = df.groupby('away_team').apply(calculate_form).reset_index()
            away_form.columns = ['team', 'away_form_avg']
            
            # Form farkı
            df = df.merge(home_form, on='home_team', how='left')
            df = df.merge(away_form, on='away_team', how='left')
            df['form_diff'] = df['home_form_avg'] - df['away_form_avg']
        
        # Motivasyon faktörleri
        if 'home_motivation' in df.columns and 'away_motivation' in df.columns:
            df['motivation_diff'] = df['home_motivation'] - df['away_motivation']
        
        # Güç farkı
        if 'home_strength' in df.columns and 'away_strength' in df.columns:
            df['strength_diff'] = (df['home_strength'] - df['away_strength']) / 100
        
        return df
    
    def calculate_xg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        xG tabanlı özellikler oluştur
        
        Args:
            df: Maç verileri DataFrame'i
            
        Returns:
            xG özellikleri eklenmiş DataFrame
        """
        logger.info("xG tabanlı özellikler oluşturuluyor...")
        
        # xG farkı
        df['xg_diff'] = df['home_xg'] - df['away_xg']
        
        # xG toplamı
        df['total_xg'] = df['home_xg'] + df['away_xg']
        
        # xG oranı
        df['home_xg_ratio'] = df['home_xg'] / (df['home_xg'] + df['away_xg']).replace(0, 1)
        df['away_xg_ratio'] = df['away_xg'] / (df['home_xg'] + df['away_xg']).replace(0, 1)
        
        # xG verimliliği (gerçek gol / xG)
        df['home_xg_efficiency'] = df['home_score'] / df['home_xg'].replace(0, 1)
        df['away_xg_efficiency'] = df['away_score'] / df['away_xg'].replace(0, 1)
        
        return df
    
    def calculate_possession_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Top hakimiyeti tabanlı özellikler oluştur
        
        Args:
            df: Maç verileri DataFrame'i
            
        Returns:
            Top hakimiyeti özellikleri eklenmiş DataFrame
        """
        logger.info("Top hakimiyeti özellikleri oluşturuluyor...")
        
        # Top hakimiyeti farkı
        df['possession_diff'] = df['home_possession'] - df['away_possession']
        
        # Top hakimiyeti oranı
        df['home_possession_ratio'] = df['home_possession'] / 100
        df['away_possession_ratio'] = df['away_possession'] / 100
        
        return df
    
    def calculate_shot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Şut tabanlı özellikler oluştur
        
        Args:
            df: Maç verileri DataFrame'i
            
        Returns:
            Şut özellikleri eklenmiş DataFrame
        """
        logger.info("Şut özellikleri oluşturuluyor...")
        
        # Toplam şut
        df['total_shots'] = df['home_shots'] + df['away_shots']
        
        # Şut farkı
        df['shot_diff'] = df['home_shots'] - df['away_shots']
        
        # İsabetli şut oranı
        df['home_shot_accuracy'] = df['home_shots_on_target'] / df['home_shots'].replace(0, 1)
        df['away_shot_accuracy'] = df['away_shots_on_target'] / df['away_shots'].replace(0, 1)
        
        # Şut verimliliği (gol / şut)
        df['home_shot_efficiency'] = df['home_score'] / df['home_shots'].replace(0, 1)
        df['away_shot_efficiency'] = df['away_score'] / df['away_shots'].replace(0, 1)
        
        return df
    
    def calculate_discipline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Disiplin tabanlı özellikler oluştur
        
        Args:
            df: Maç verileri DataFrame'i
            
        Returns:
            Disiplin özellikleri eklenmiş DataFrame
        """
        logger.info("Disiplin özellikleri oluşturuluyor...")
        
        # Toplam faul
        df['total_fouls'] = df['home_fouls'] + df['away_fouls']
        
        # Faul farkı
        df['foul_diff'] = df['home_fouls'] - df['away_fouls']
        
        # Toplam kart
        df['total_yellow_cards'] = df['home_yellow_cards'] + df['away_yellow_cards']
        df['total_red_cards'] = df['home_red_cards'] + df['away_red_cards']
        
        # Kart farkı
        df['yellow_card_diff'] = df['home_yellow_cards'] - df['away_yellow_cards']
        df['red_card_diff'] = df['home_red_cards'] - df['away_red_cards']
        
        return df
    
    def calculate_team_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takım istatistiklerini hesapla
        
        Args:
            df: Maç verileri DataFrame'i
            
        Returns:
            Takım istatistikleri eklenmiş DataFrame
        """
        logger.info("Takım istatistikleri hesaplanıyor...")
        
        teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        
        # Her takım için istatistikler
        team_stats = {}
        
        for team in teams:
            team_matches = df[
                (df['home_team'] == team) | (df['away_team'] == team)
            ]
            
            if len(team_matches) == 0:
                continue
            
            # Ev sahibi istatistikleri
            home_matches = team_matches[team_matches['home_team'] == team]
            away_matches = team_matches[team_matches['away_team'] == team]
            
            stats = {
                'home_avg_xg': home_matches['home_xg'].mean() if len(home_matches) > 0 else 0,
                'away_avg_xg': away_matches['away_xg'].mean() if len(away_matches) > 0 else 0,
                'home_avg_possession': home_matches['home_possession'].mean() if len(home_matches) > 0 else 50,
                'away_avg_possession': away_matches['away_possession'].mean() if len(away_matches) > 0 else 50,
                'home_avg_shots': home_matches['home_shots'].mean() if len(home_matches) > 0 else 0,
                'away_avg_shots': away_matches['away_shots'].mean() if len(away_matches) > 0 else 0,
                'home_win_rate': (home_matches['result'] == 'Home').mean() if len(home_matches) > 0 else 0,
                'away_win_rate': (away_matches['result'] == 'Away').mean() if len(away_matches) > 0 else 0,
                'total_matches': len(team_matches)
            }
            
            team_stats[team] = stats
        
        # İstatistikleri DataFrame'e ekle
        for stat_name in ['home_avg_xg', 'away_avg_xg', 'home_avg_possession', 'away_avg_possession',
                         'home_avg_shots', 'away_avg_shots', 'home_win_rate', 'away_win_rate']:
            df[f'home_{stat_name}'] = df['home_team'].map({team: stats[stat_name] for team, stats in team_stats.items()})
            df[f'away_{stat_name}'] = df['away_team'].map({team: stats[stat_name] for team, stats in team_stats.items()})
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Etkileşim özellikleri oluştur
        
        Args:
            df: Maç verileri DataFrame'i
            
        Returns:
            Etkileşim özellikleri eklenmiş DataFrame
        """
        logger.info("Etkileşim özellikleri oluşturuluyor...")
        
        # Form etkileşimi
        df['form_interaction'] = df['home_form'] * df['away_form']
        df['form_diff'] = df['home_form'] - df['away_form']
        
        # xG etkileşimi
        df['xg_form_interaction'] = df['home_xg'] * df['home_form'] - df['away_xg'] * df['away_form']
        
        # Top hakimiyeti etkileşimi
        df['possession_form_interaction'] = df['home_possession'] * df['home_form'] - df['away_possession'] * df['away_form']
        
        # Şut etkileşimi
        df['shot_form_interaction'] = df['home_shots'] * df['home_form'] - df['away_shots'] * df['away_form']
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hedef değişkeni oluştur
        
        Args:
            df: Maç verileri DataFrame'i
            
        Returns:
            Hedef değişkeni eklenmiş DataFrame
        """
        logger.info("Hedef değişkeni oluşturuluyor...")
        
        # Sonuç sütunu zaten var, sadece sayısal hale getir
        result_mapping = {'Home': 0, 'Draw': 1, 'Away': 2}
        df['target'] = df['result'].map(result_mapping)
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm özellikleri oluştur
        """
        logger.info("Tüm özellikler oluşturuluyor...")
        
        # 1. Takım form özellikleri
        df = self.calculate_team_form(df)
        
        # 2. Maç bağlamı özellikleri
        df = self.calculate_match_context_features(df)
        
        # 3. xG tabanlı özellikler
        df = self.calculate_xg_features(df)
        
        # 4. Top hakimiyeti özellikleri
        df = self.calculate_possession_features(df)
        
        # 5. Şut özellikleri
        df = self.calculate_shot_features(df)
        
        # 6. Disiplin özellikleri
        df = self.calculate_discipline_features(df)
        
        # 7. Takım istatistikleri
        df = self.calculate_team_statistics(df)
        
        # 8. Etkileşim özellikleri
        df = self.create_interaction_features(df)
        
        # 9. Gelişmiş özellikler
        df = self.calculate_advanced_features(df)
        
        # 10. Hedef değişkeni oluştur
        df = self.create_target_variable(df)
        
        logger.info(f"Toplam {len(df.columns)} özellik oluşturuldu")
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Özellik önem sıralaması oluştur
        
        Args:
            df: Özellikler eklenmiş DataFrame
            
        Returns:
            Özellik önem sıralaması
        """
        # Hedef değişkeni hariç tut
        feature_cols = [col for col in df.columns if col not in ['match_id', 'date', 'home_team', 'away_team', 'result', 'target']]
        
        # Korelasyon hesapla
        correlations = df[feature_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
        
        # DataFrame'e çevir
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        return importance_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_matches.csv"):
        """
        İşlenmiş veriyi kaydet
        
        Args:
            df: İşlenmiş DataFrame
            filename: Dosya adı
        """
        output_path = f"{self.processed_dir}/{filename}"
        df.to_csv(output_path, index=False)
        logger.info(f"İşlenmiş veri kaydedildi: {output_path}")
    
    def process_all_data(self) -> pd.DataFrame:
        """
        Tüm veriyi işle ve özellikler oluştur
        
        Returns:
            İşlenmiş veri DataFrame'i
        """
        logger.info("Tüm veri işleme süreci başlatılıyor...")
        
        # Ham verileri yükle
        raw_data = self.load_raw_data()
        
        if raw_data['matches'].empty:
            logger.error("Maç verileri bulunamadı!")
            return pd.DataFrame()
        
        # Özellikleri oluştur
        processed_df = self.engineer_all_features(raw_data['matches'].copy())
        
        # İşlenmiş veriyi kaydet
        self.save_processed_data(processed_df)
        
        # Özellik önem sıralamasını oluştur
        importance_df = self.get_feature_importance_ranking(processed_df)
        importance_df.to_csv(f"{self.processed_dir}/feature_importance.csv", index=False)
        
        logger.info("Veri işleme tamamlandı!")
        return processed_df

    def calculate_match_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maç bağlamı özelliklerini hesapla
        """
        logger.info("Maç bağlamı özellikleri oluşturuluyor...")
        
        # Tarih özellikleri
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # Sezon dönemi özellikleri
        df['is_season_start'] = df['month'].isin([2, 3]).astype(int)
        df['is_season_end'] = df['month'].isin([10, 11]).astype(int)
        df['is_playoff_season'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # Derbi özellikleri
        derby_pairs = [
            ('LA Galaxy', 'Los Angeles FC'),
            ('New York City FC', 'New York Red Bulls'),
            ('Chicago Fire', 'St. Louis CITY SC')
        ]
        
        def is_derby(row):
            for team1, team2 in derby_pairs:
                if (row['home_team'] == team1 and row['away_team'] == team2) or \
                   (row['home_team'] == team2 and row['away_team'] == team1):
                    return 1
            return 0
        
        df['is_derby'] = df.apply(is_derby, axis=1)
        
        # Takım güç kategorileri
        if 'home_strength' in df.columns and 'away_strength' in df.columns:
            # Güçlü takım (>80), Orta takım (60-80), Zayıf takım (<60)
            df['home_strength_category'] = pd.cut(df['home_strength'], 
                                                bins=[0, 60, 80, 100], 
                                                labels=['Weak', 'Medium', 'Strong'])
            df['away_strength_category'] = pd.cut(df['away_strength'], 
                                                bins=[0, 60, 80, 100], 
                                                labels=['Weak', 'Medium', 'Strong'])
            
            # Güç kategorisi farkı
            strength_cat_map = {'Weak': 0, 'Medium': 1, 'Strong': 2}
            df['home_strength_cat'] = df['home_strength_category'].map(strength_cat_map)
            df['away_strength_cat'] = df['away_strength_category'].map(strength_cat_map)
            df['strength_cat_diff'] = df['home_strength_cat'] - df['away_strength_cat']
        
        return df
    
    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş özellikler oluştur
        """
        logger.info("Gelişmiş özellikler oluşturuluyor...")
        
        # Toplam faktör skoru
        if all(col in df.columns for col in ['playoff_bonus', 'derby_factor', 'weekend_factor', 'season_factor']):
            df['total_context_factor'] = df['playoff_bonus'] * df['derby_factor'] * df['weekend_factor'] * df['season_factor']
        
        # Güç ve form kombinasyonu
        if all(col in df.columns for col in ['strength_diff', 'form_diff', 'motivation_diff']):
            df['overall_advantage'] = (df['strength_diff'] * 0.4 + 
                                     df['form_diff'] * 0.3 + 
                                     df['motivation_diff'] * 0.3)
        
        # xG verimliliği
        if all(col in df.columns for col in ['home_xg', 'away_xg', 'home_score', 'away_score']):
            df['home_xg_efficiency'] = df['home_score'] / df['home_xg'].replace(0, 1)
            df['away_xg_efficiency'] = df['away_score'] / df['away_xg'].replace(0, 1)
            df['xg_efficiency_diff'] = df['home_xg_efficiency'] - df['away_xg_efficiency']
        
        # Şut verimliliği
        if all(col in df.columns for col in ['home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target']):
            df['home_shot_accuracy'] = df['home_shots_on_target'] / df['home_shots'].replace(0, 1)
            df['away_shot_accuracy'] = df['away_shots_on_target'] / df['away_shots'].replace(0, 1)
            df['shot_accuracy_diff'] = df['home_shot_accuracy'] - df['away_shot_accuracy']
        
        # Disiplin faktörü
        if all(col in df.columns for col in ['home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards']):
            df['home_discipline_score'] = -(df['home_yellow_cards'] * 0.5 + df['home_red_cards'] * 2)
            df['away_discipline_score'] = -(df['away_yellow_cards'] * 0.5 + df['away_red_cards'] * 2)
            df['discipline_diff'] = df['home_discipline_score'] - df['away_discipline_score']
        
        return df

def main():
    """Ana fonksiyon - özellik mühendisliği işlemini başlatır"""
    engineer = MLSFeatureEngineer()
    
    # Tüm veriyi işle
    processed_df = engineer.process_all_data()
    
    if not processed_df.empty:
        print("\n" + "="*50)
        print("ÖZELLİK MÜHENDİSLİĞİ ÖZETİ")
        print("="*50)
        print(f"Toplam satır: {len(processed_df)}")
        print(f"Toplam özellik: {len(processed_df.columns)}")
        print(f"Hedef değişken dağılımı:")
        print(processed_df['target'].value_counts())
        
        # En önemli özellikleri göster
        importance_df = engineer.get_feature_importance_ranking(processed_df)
        print(f"\nEn önemli 10 özellik:")
        print(importance_df.head(10))
    
    print("\n" + "="*50)
    print("Özellik mühendisliği tamamlandı! 🔧")

if __name__ == "__main__":
    main() 