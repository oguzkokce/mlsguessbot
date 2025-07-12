# ⚽ MLS Maç Tahmin Sistemi

Major League Soccer (MLS) maçlarının sonuçlarını tahmin eden makine öğrenmesi tabanlı web uygulaması.

## 🚀 Canlı Demo

[Streamlit Cloud'ta Canlı Demo](https://mls-prediction-app.streamlit.app)

## 📊 Özellikler

- **Gerçek Zamanlı Tahminler**: Ev sahibi vs deplasman takımı maç sonuçları
- **Gelişmiş İstatistikler**: xG, form, saha avantajı, takım dengesi
- **Görsel Analizler**: Aylık sonuçlar, takım performansları
- **Model Performansı**: Accuracy, F1 Score, Confusion Matrix

## 🛠️ Teknolojiler

- **Python 3.12**
- **Streamlit** - Web arayüzü
- **Scikit-learn** - Makine öğrenmesi
- **XGBoost** - Gradient boosting
- **Plotly** - Veri görselleştirme
- **Pandas** - Veri işleme

## 📈 Model Özellikleri

- **80+ Özellik**: xG, form, saha avantajı, takım dengesi
- **Çoklu Model**: RandomForest, XGBoost, LogisticRegression
- **Hiperparametre Optimizasyonu**: GridSearchCV ile
- **Gerçekçi Veri**: 2024/2025 sezonu simülasyonu

## 🚀 Yerel Kurulum

```bash
# Repository'yi klonla
git clone https://github.com/username/mls-prediction-app.git
cd mls-prediction-app

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Veri işleme ve model eğitimi
python data_loader.py
python feature_engineering.py
python train_model.py

# Uygulamayı çalıştır
streamlit run app.py
```

## 🌐 Ücretsiz Yayınlama

### 1. Streamlit Cloud (Önerilen)

1. **GitHub'a Yükle**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/username/mls-prediction-app.git
   git push -u origin main
   ```

2. **Streamlit Cloud'ta Yayınla**:
   - [share.streamlit.io](https://share.streamlit.io) adresine git
   - GitHub hesabınızla giriş yap
   - Repository'yi seç
   - "Deploy" butonuna tıkla

### 2. Vercel (Alternatif)

1. **Vercel CLI Kur**:
   ```bash
   npm i -g vercel
   ```

2. **vercel.json Oluştur**:
   ```json
   {
     "builds": [
       {
         "src": "app.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "app.py"
       }
     ]
   }
   ```

3. **Yayınla**:
   ```bash
   vercel
   ```

### 3. Heroku (Alternatif)

1. **Procfile Oluştur**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Heroku CLI ile Yayınla**:
   ```bash
   heroku create mls-prediction-app
   git push heroku main
   ```

## 📁 Proje Yapısı

```
mls/
├── app.py                 # Ana Streamlit uygulaması
├── data_loader.py         # Veri yükleme ve simülasyon
├── feature_engineering.py # Özellik mühendisliği
├── train_model.py         # Model eğitimi
├── requirements.txt       # Python bağımlılıkları
├── .streamlit/           # Streamlit konfigürasyonu
│   └── config.toml
├── data/                 # Veri dosyaları
│   ├── raw/
│   ├── processed/
│   └── models/
└── README.md
```

## 🔧 Konfigürasyon

### Streamlit Konfigürasyonu (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

## 📊 Veri Kaynakları

- **FBref.com**: xG, xA, pas haritaları
- **MLSPA**: Oyuncu maaşları
- **Kaggle MLS Dataset**: Maç geçmişi
- **Simülasyon**: 2024/2025 sezonu gerçekçi veri

## 🤝 Katkıda Bulunma

1. Fork yap
2. Feature branch oluştur (`git checkout -b feature/amazing-feature`)
3. Commit yap (`git commit -m 'Add amazing feature'`)
4. Push yap (`git push origin feature/amazing-feature`)
5. Pull Request aç

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## ⚠️ Önemli Not

Bu proje eğitim amaçlı geliştirilmiştir. Gerçek bahis oyunları için kullanılması önerilmez.

## 📞 İletişim

- **GitHub**: [@username](https://github.com/username)
- **Email**: your.email@example.com

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! 