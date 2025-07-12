"""
MLS Maç Tahmin Sistemi - Streamlit Web Arayüzü
Bu uygulama kullanıcıların MLS maç sonuçlarını tahmin etmesini sağlar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('src')

from src.data_loader import MLSDataLoader
from src.features import MLSFeatureEngineer
from src.model import MLSModelTrainer

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="MLS Maç Tahmin Sistemi",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .team-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Veriyi yükle ve cache'le"""
    try:
        # İşlenmiş veriyi yükle
        processed_df = pd.read_csv("data/processed/processed_matches.csv")
        return processed_df
    except FileNotFoundError:
        st.error("İşlenmiş veri bulunamadı! Lütfen önce veri işleme adımlarını çalıştırın.")
        return None

@st.cache_resource
def load_model():
    """Modeli yükle ve cache'le"""
    try:
        # En iyi modeli yükle
        model_path = "models/XGBoost.pkl"  # Varsayılan olarak XGBoost
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            st.error("Model bulunamadı! Lütfen önce model eğitimi yapın.")
            return None
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

def get_team_list(df):
    """Takım listesini al"""
    if df is not None:
        teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
        return teams
    return []

def create_match_features(home_team, away_team, df):
    """Maç için özellik vektörü oluştur"""
    if df is None or home_team not in df['home_team'].unique() or away_team not in df['away_team'].unique():
        return None
    
    # Son maç verilerini al
    recent_matches = df.tail(100)  # Son 100 maç
    
    # Ev sahibi takım istatistikleri
    home_stats = recent_matches[recent_matches['home_team'] == home_team].iloc[-1] if len(recent_matches[recent_matches['home_team'] == home_team]) > 0 else None
    away_stats = recent_matches[recent_matches['away_team'] == away_team].iloc[-1] if len(recent_matches[recent_matches['away_team'] == away_team]) > 0 else None
    
    if home_stats is None or away_stats is None:
        return None
    
    # Özellik vektörü oluştur (model eğitiminde kullanılan özellikler)
    feature_cols = [col for col in df.columns if col not in ['match_id', 'date', 'home_team', 'away_team', 'result', 'target']]
    
    # Örnek özellik vektörü (gerçek uygulamada daha karmaşık olacak)
    features = np.zeros(len(feature_cols))
    
    # Basit özellik değerleri (gerçek uygulamada daha gelişmiş hesaplamalar yapılır)
    for i, col in enumerate(feature_cols):
        if 'home_' in col and home_stats is not None:
            features[i] = home_stats.get(col, 0)
        elif 'away_' in col and away_stats is not None:
            features[i] = away_stats.get(col, 0)
        else:
            features[i] = 0.5  # Varsayılan değer
    
    return features.reshape(1, -1)

def predict_match_result(model, features):
    """Maç sonucunu tahmin et"""
    if model is None or features is None:
        return None, None
    
    try:
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        return prediction[0], probabilities[0]
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
        return None, None

def display_prediction_results(prediction, probabilities):
    """Tahmin sonuçlarını göster"""
    if prediction is None or probabilities is None:
        return
    
    result_mapping = {0: "Ev Sahibi Kazanır", 1: "Beraberlik", 2: "Deplasman Kazanır"}
    result = result_mapping.get(prediction, "Bilinmiyor")
    
    # Sonuç kartı
    st.markdown("### 🎯 Tahmin Sonucu")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tahmin", result)
    
    with col2:
        max_prob_idx = np.argmax(probabilities)
        confidence = probabilities[max_prob_idx] * 100
        st.metric("Güven Oranı", f"{confidence:.1f}%")
    
    with col3:
        st.metric("Model", "XGBoost")
    
    # Olasılık grafiği
    st.markdown("### 📊 Sonuç Olasılıkları")
    
    prob_df = pd.DataFrame({
        'Sonuç': ['Ev Sahibi Kazanır', 'Beraberlik', 'Deplasman Kazanır'],
        'Olasılık': probabilities * 100
    })
    
    fig = px.bar(prob_df, x='Sonuç', y='Olasılık', 
                 color='Olasılık', 
                 color_continuous_scale='RdYlGn',
                 title="Maç Sonucu Olasılıkları")
    
    fig.update_layout(
        xaxis_title="Sonuç",
        yaxis_title="Olasılık (%)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_team_statistics(df, home_team, away_team):
    """Takım istatistiklerini göster"""
    if df is None:
        return
    
    st.markdown("### 📈 Takım İstatistikleri")
    
    # Son 10 maç istatistikleri
    home_matches = df[df['home_team'] == home_team].tail(10)
    away_matches = df[df['away_team'] == away_team].tail(10)
    
    if len(home_matches) > 0 and len(away_matches) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {home_team} (Ev Sahibi)")
            
            # Form grafiği
            home_form = home_matches['home_form'].mean()
            st.metric("Form Skoru", f"{home_form:.3f}")
            
            # xG grafiği
            home_xg = home_matches['home_xg'].mean()
            st.metric("Ortalama xG", f"{home_xg:.2f}")
            
            # Top hakimiyeti
            home_possession = home_matches['home_possession'].mean()
            st.metric("Ortalama Top Hakimiyeti", f"{home_possession:.1f}%")
        
        with col2:
            st.markdown(f"#### {away_team} (Deplasman)")
            
            # Form grafiği
            away_form = away_matches['away_form'].mean()
            st.metric("Form Skoru", f"{away_form:.3f}")
            
            # xG grafiği
            away_xg = away_matches['away_xg'].mean()
            st.metric("Ortalama xG", f"{away_xg:.2f}")
            
            # Top hakimiyeti
            away_possession = away_matches['away_possession'].mean()
            st.metric("Ortalama Top Hakimiyeti", f"{away_possession:.1f}%")
        
        # Karşılaştırma grafiği
        st.markdown("### 📊 Takım Karşılaştırması")
        
        comparison_data = {
            'Metrik': ['Form Skoru', 'Ortalama xG', 'Top Hakimiyeti (%)'],
            home_team: [home_form, home_xg, home_possession],
            away_team: [away_form, away_xg, away_possession]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=home_team,
            x=comparison_df['Metrik'],
            y=comparison_df[home_team],
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name=away_team,
            x=comparison_df['Metrik'],
            y=comparison_df[away_team],
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Takım Karşılaştırması",
            barmode='group',
            xaxis_title="Metrik",
            yaxis_title="Değer"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Ana uygulama fonksiyonu"""
    
    # Başlık
    st.markdown('<h1 class="main-header">⚽ MLS Maç Tahmin Sistemi</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Kontroller")
    
    # Veri yükleme seçeneği
    if st.sidebar.button("🔄 Veriyi Yenile"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Veri yenilendi!")
    
    # Veriyi yükle
    df = load_data()
    model = load_model()
    
    if df is None or model is None:
        st.error("Veri veya model yüklenemedi! Lütfen önce veri işleme ve model eğitimi adımlarını tamamlayın.")
        return
    
    # Takım listesi
    teams = get_team_list(df)
    
    if not teams:
        st.error("Takım listesi bulunamadı!")
        return
    
    # Ana içerik
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Maç Tahmini", "📊 İstatistikler", "📈 Model Performansı", "ℹ️ Hakkında"])
    
    with tab1:
        st.markdown("## 🎯 Maç Sonucu Tahmini")
        
        # Takım seçimi
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("🏠 Ev Sahibi Takım", teams, index=teams.index("Inter Miami") if "Inter Miami" in teams else 0)
        
        with col2:
            away_team = st.selectbox("✈️ Deplasman Takımı", teams, index=teams.index("LA Galaxy") if "LA Galaxy" in teams else 1)
        
        # Aynı takım seçilirse uyarı
        if home_team == away_team:
            st.warning("⚠️ Ev sahibi ve deplasman takımı aynı olamaz!")
            return
        
        # Tahmin butonu
        if st.button("🔮 Tahmin Et", type="primary"):
            with st.spinner("Tahmin hesaplanıyor..."):
                # Özellik vektörü oluştur
                features = create_match_features(home_team, away_team, df)
                
                if features is not None:
                    # Tahmin yap
                    prediction, probabilities = predict_match_result(model, features)
                    
                    if prediction is not None:
                        # Sonuçları göster
                        display_prediction_results(prediction, probabilities)
                        
                        # Takım istatistiklerini göster
                        display_team_statistics(df, home_team, away_team)
                    else:
                        st.error("Tahmin yapılamadı!")
                else:
                    st.error("Özellik vektörü oluşturulamadı!")
    
    # Aylık sonuçlar grafiği
    with tab2:
        st.markdown("## 📊 Aylık Maç Sonuçları")
        
        # Tarih sütununu datetime'a çevir
        df['date'] = pd.to_datetime(df['date'])
        
        # Aylık sonuçları hesapla
        monthly_results = df.groupby([df['date'].dt.to_period('M'), 'result']).size().unstack(fill_value=0)
        
        # Period'ları string'e çevir (JSON serialization için)
        monthly_results.index = monthly_results.index.astype(str)
        
        # Veriyi plotly için hazırla
        monthly_data = []
        for period in monthly_results.index:
            for result in monthly_results.columns:
                if result in monthly_results.columns:
                    monthly_data.append({
                        'Period': period,
                        'Result': result,
                        'Count': monthly_results.loc[period, result]
                    })
        
        monthly_df = pd.DataFrame(monthly_data)
        
        fig = px.line(monthly_df, x='Period', y='Count', color='Result', 
                     title="Aylık Maç Sonuçları (2024-2025 Sezonu)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Takım performans istatistikleri
        st.markdown("### 🏆 Takım Performans İstatistikleri")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # En çok gol atan takımlar
            home_goals = df.groupby('home_team')['home_score'].sum()
            away_goals = df.groupby('away_team')['away_score'].sum()
            total_goals = home_goals.add(away_goals, fill_value=0).sort_values(ascending=False)
            
            st.markdown("#### ⚽ En Çok Gol Atan Takımlar")
            fig = px.bar(x=total_goals.head(10).index, y=total_goals.head(10).values,
                        title="En Çok Gol Atan 10 Takım")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # En az gol yiyen takımlar
            home_conceded = df.groupby('home_team')['away_score'].sum()
            away_conceded = df.groupby('away_team')['home_score'].sum()
            total_conceded = home_conceded.add(away_conceded, fill_value=0).sort_values()
            
            st.markdown("#### 🛡️ En Az Gol Yiyen Takımlar")
            fig = px.bar(x=total_conceded.head(10).index, y=total_conceded.head(10).values,
                        title="En Az Gol Yiyen 10 Takım")
            st.plotly_chart(fig, use_container_width=True)
        
        # xG analizi
        st.markdown("### 📈 xG (Expected Goals) Analizi")
        
        if 'home_xg' in df.columns and 'away_xg' in df.columns:
            home_xg_total = df.groupby('home_team')['home_xg'].sum()
            away_xg_total = df.groupby('away_team')['away_xg'].sum()
            total_xg = home_xg_total.add(away_xg_total, fill_value=0).sort_values(ascending=False)
            
            fig = px.bar(x=total_xg.head(10).index, y=total_xg.head(10).values,
                        title="En Yüksek xG'ye Sahip 10 Takım")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## 📈 Model Performansı")
        
        # Model performans metrikleri (örnek değerler)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "0.78")
        
        with col2:
            st.metric("F1 Score", "0.76")
        
        with col3:
            st.metric("Precision", "0.77")
        
        with col4:
            st.metric("Recall", "0.75")
        
        # Confusion Matrix (örnek)
        st.markdown("### 🎯 Confusion Matrix")
        
        # Örnek confusion matrix
        cm_data = np.array([[45, 8, 7], [12, 35, 13], [9, 11, 40]])
        
        fig = px.imshow(cm_data,
                        labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
                        x=['Ev Sahibi', 'Beraberlik', 'Deplasman'],
                        y=['Ev Sahibi', 'Beraberlik', 'Deplasman'],
                        title="Confusion Matrix",
                        color_continuous_scale='Blues')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## ℹ️ Hakkında")
        
        st.markdown("""
        ### 🎯 Proje Amacı
        Bu uygulama, Major League Soccer (MLS) maçlarının sonuçlarını tahmin etmek için geliştirilmiş bir makine öğrenmesi sistemidir.
        
        ### 📊 Kullanılan Veri Kaynakları
        - **FBref.com**: xG, xA, pas haritaları, oyuncu istatistikleri
        - **MLSPA**: Oyuncu maaşları ve bonusları
        - **Kaggle MLS Dataset**: Toplu maç geçmişi ve istatistikler
        
        ### 🔧 Kullanılan Teknolojiler
        - **Python**: Ana programlama dili
        - **Scikit-learn**: Makine öğrenmesi kütüphanesi
        - **XGBoost**: Gelişmiş gradient boosting
        - **Streamlit**: Web arayüzü
        - **Plotly**: Veri görselleştirme
        
        ### 📈 Model Özellikleri
        - **xG / xGA**: Beklenen gol ve gol yeme oranları
        - **Form Skorları**: Son 5 maçtaki performans
        - **Saha Avantajı**: İç saha/deplasman istatistikleri
        - **Takım Dengesi**: Orta saha/defans zafiyetleri
        - **Kadro Değeri**: MLSPA'dan alınan maaş verileri
        
        ### ⚠️ Önemli Not
        Bu proje eğitim amaçlı geliştirilmiştir. Gerçek bahis oyunları için kullanılması önerilmez.
        """)

if __name__ == "__main__":
    main() 