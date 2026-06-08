import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Sayfa Genişlik Ayarları
st.set_page_config(page_title="Küresel Çapraz Kur Projeksiyon Platformu", layout="wide")

# --- BAŞLIK ALANI ---
st.title("🏛️ Küresel Döviz ve Çapraz Kur Tahmin Platformu")
st.caption("Bulut Tabanlı, Yapay Zeka Destekli Anlık Finansal Projeksiyon Yazılımı")
st.markdown("---")

# Model ve Scaler Dosyalarını Yükleme (Arka Plan Güvencesi)
@st.cache_resource
def load_assets():
    try:
        with open('doviz_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- PARA BİRİMİ SEÇİM SÖZLÜĞÜ (YAHOO FINANCE KODLARI) ---
# Dünyadaki en popüler para birimleri ve Yahoo'daki kısa kodları
para_birimleri = {
    "Türk Lirası (TRY)": "TRY",
    "Amerikan Doları (USD)": "USD",
    "Euro (EUR)": "EUR",
    "İngiliz Sterlini (GBP)": "GBP",
    "İsviçre Frangı (CHF)": "CHF",
    "Kuveyt Dinarı (KWD)": "KWD",
    "Japon Yeni (JPY)": "JPY",
    "Suudi Arabistan Riyali (SAR)": "SAR"
}

# --- 1. AŞAMA: KULLANICI GİRİŞ PANELİ (AÇILIŞ EKRANI) ---
st.subheader("🎯 Projeksiyon Başlatma Paneli")
st.info("Lütfen analiz etmek istediğiniz parite çiftini seçiniz. Sistem bulut üzerinden anlık verileri çekerek hesaplama yapacaktır.")

col_input1, col_input2 = st.columns(2)

with col_input1:
    baz_para = st.selectbox("1. Baz Para Birimi (Satın Alınacak / Tahmin Edilecek):", list(para_birimleri.keys()), index=1) # Varsayılan USD

with col_input2:
    karsi_para = st.selectbox("2. Karşı Para Birimi (Ödeme Yapılacak Cins):", list(para_birimleri.keys()), index=0) # Varsayılan TRY

# "Analiz Et ve Hesapla" Butonu
st.markdown("<br>", unsafe_allow_html=True)
analiz_butonu = st.button("🚀 BULUTTAN VERİLERİ ÇEK VE YAPAY ZEKAYI ÇALIŞTIR", use_container_width=True)

# --- 2. AŞAMA: HESAPLAMA VE GÖRÜNTÜLEME MOTORU ---
# Kullanıcı butona bastığında veya pariteler seçildiğinde tetiklenir
if analiz_butonu or 'analiz_yapildi' in st.session_state:
    st.session_state['analiz_yapildi'] = True # Sayfa yenilendiğinde seçimin kaybolmaması için state tutuyoruz
    
    kod_baz = para_birimleri[baz_para]
    kod_karsi = para_birimleri[karsi_para]
    
    # Yahoo Finance Çapraz Kur Kuralı:
    # Eğer baz para USD ise direkt 'TRY=X' veya 'EUR=X' olur.
    # Eğer baz para USD değilse 'EURUSD=X' veya 'EURTRY=X' şeklinde birleşir.
    if kod_baz == kod_karsi:
        st.error("❌ Aynı para birimleri arasında çapraz kur analizi yapılamaz! Lütfen farklı iki para birimi seçiniz.")
    else:
        if kod_baz == "USD":
            ticker = f"{kod_karsi}=X"
            ters_cevir = False
        elif kod_karsi == "USD":
            ticker = f"{kod_baz}=X"
            ters_cevir = True # Veri çakışmasını önlemek için matematiksel doğrulamada kullanacağız
        else:
            ticker = f"{kod_baz}{kod_karsi}=X"
            ters_cevir = False

        # Buluttan canlı verileri çekme aşaması
        with st.spinner("⏳ Yahoo Finance bulut sunucularına bağlanılıyor, anlık veriler çekiliyor..."):
            try:
                data = yf.download(ticker, period="3y", interval="1d")
                data = data[['Close']].dropna()
                
                # Eğer USD bazlı parite ters çekildiyse matematiksel olarak 1/oran yapıyoruz (Çapraz kur standartı)
                if ters_cevir:
                    data['Close'] = 1 / data['Close']
            except Exception as e:
                st.error(f"❌ Buluttan veri çekme hatası oluştu: {e}")
                data = pd.DataFrame()

        if not data.empty:
            # Cari fiyatları sabitleme
            guncel_fiyat = float(data['Close'].iloc[-1].values[0]) if hasattr(data['Close'].iloc[-1], 'values') else float(data['Close'].iloc[-1])
            onceki_fiyat = float(data['Close'].iloc[-2].values[0]) if hasattr(data['Close'].iloc[-2], 'values') else float(data['Close'].iloc[-2])
            degisim = guncel_fiyat - onceki_fiyat

            st.markdown("---")
            # --- ÜST METRİK KUTULARI ---
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(label=f"Anlık Cari Kur (1 {kod_baz} kaç {kod_karsi}?)", value=f"{guncel_fiyat:.4f} {kod_karsi}", delta=f"{degisim:.4f} {kod_karsi}")
            
            # --- ZİNCİRLEME GELECEK TAHMİN MOTORU ---
            prediction_days = 60
            gecmis_seri = list(data['Close'][-prediction_days:].values.flatten())
            son_30_gun = data['Close'][-30:].values.flatten()
            trend_gunluk = np.mean(np.diff(son_30_gun))

            # 1 Aylık Günlük Tahmin Döngüsü
            günlük_tarihler = []
            günlük_tahminler = []
            son_tarih_gun = data.index[-1]

            for i in range(1, 31):
                if model and scaler and kod_karsi == "TRY" and kod_baz == "USD": # Ana model sadece USDTRY için tam oturur
                    input_window = np.array(gecmis_seri[-prediction_days:]).reshape(1, -1)
                    input_scaled = scaler.transform(input_window.T).T
                    pred_scaled = model.predict(input_scaled)
                    ham_tahmin = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                    
                    referans = gecmis_seri[-1]
                    if ham_tahmin < (referans * 0.98) or ham_tahmin > (referans * 1.02):
                        tahmin_adim = referans + trend_gunluk
                    else:
                        tahmin_adim = (ham_tahmin * 0.3) + ((referans + trend_gunluk) * 0.7)
                else:
                    # Diğer tüm küresel çapraz kurlarda bulut verisinin momentum trendini çalıştırır
                    tahmin_adim = gecmis_seri[-1] + trend_gunluk
                    
                günlük_tahminler.append(tahmin_adim)
                gecmis_seri.append(tahmin_adim)
                son_tarih_gun += timedelta(days=1)
                günlük_tarihler.append(son_tarih_gun)

            # 1 Yıllık Aylık Tahmin Döngüsü
            aylik_tarihler = [günlük_tarihler[-1]]
            aylik_tahminler = [günlük_tahminler[-1]]
            son_tarih_ay = günlük_tarihler[-1]

            for ay in range(1, 12):
                for gun in range(30):
                    tahmin_adim = gecmis_seri[-1] + trend_gunluk
                    gecmis_seri.append(tahmin_adim)
                son_tarih_ay += timedelta(days=30)
                aylik_tarihler.append(son_tarih_ay)
                aylik_tahminler.append(gecmis_seri[-1])

            with col_m2:
                st.metric(label="1 Ay Sonraki Makro Öngörü", value=f"{günlük_tahminler[-1]:.4f} {kod_karsi}", delta=f"{(günlük_tahminler[-1] - guncel_fiyat):.4f} {kod_karsi}", delta_color="inverse")
            with col_m3:
                st.metric(label="1 Yıl Sonraki Sektörel Projeksiyon", value=f"{aylik_tahminler[-1]:.4f} {kod_karsi}", delta=f"{(aylik_tahminler[-1] - guncel_fiyat):.4f} {kod_karsi}", delta_color="inverse")

            # --- GRAFİKLER VE SEKMELER ---
            sekme1, sekme2 = st.tabs(["📊 Gelecek Projeksiyon Dönemleri (1 Ay & 1 Yıl)", "📈 Tarihsel Makro Geçmiş"])

            with sekme1:
                col_g1, col_g2 = st.columns([7, 3])
                
                with col_g1:
                    st.subheader(f"🎯 {kod_baz} / {kod_karsi} Gelecek Projeksiyon Haritası")
                    fig1 = go.Figure()
                    yakın_gecmis = data.tail(60)
                    # Gerçek yakın geçmiş
                    fig1.add_trace(go.Scatter(x=yakın_gecmis.index, y=yakın_gecmis['Close'].values.flatten(), mode='lines', name='Cari Grafik', line=dict(color='#00CC96', width=2)))
                    # 1 Aylık Tahmin
                    fig1.add_trace(go.Scatter(x=[data.index[-1]] + günlük_tarihler, y=[guncel_fiyat] + günlük_tahminler, mode='lines', name='1 Aylık Günlük Trend', line=dict(color='#FF5733', width=3, dash='dash')))
                    # 1 Yıllık Tahmin
                    fig1.add_trace(go.Scatter(x=aylik_tarihler, y=aylik_tahminler, mode='lines+markers', name='1 Yıllık Makro Çizgi', line=dict(color='#FFB703', width=2)))
                    
                    fig1.update_layout(xaxis_title="Tarih", yaxis_title=f"Değer ({kod_karsi})", hovermode="x unified", template="plotly_dark", height=450)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with col_g2:
                    st.subheader("📋 30 Günlük Tahmin Takvimi")
                    tablo_gun = pd.DataFrame({
                        "Tarih": [t.strftime('%Y-%m-%d') for t in günlük_tarihler],
                        f"Değer ({kod_karsi})": [f"{v:.4f}" for v in günlük_tahminler]
                    })
                    st.dataframe(tablo_gun, use_container_width=True, height=400)

            with sekme2:
                st.subheader(f"⏳ Son 3 Yıllık {kod_baz} / {kod_karsi} Grafik Geçmişi")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data.index, y=data['Close'].values.flatten(), mode='lines', name='Tarihsel Veri', line=dict(color='#636EFA', width=2)))
                fig2.update_layout(xaxis_title="Tarih", yaxis_title=f"Değer ({kod_karsi})", hovermode="x unified", template="plotly_dark", height=450)
                st.plotly_chart(fig2, use_container_width=True)
                
                st.subheader("📋 Önümüzdeki 12 Aylık Makro Dönem Kapanış Verileri")
                tablo_ay = pd.DataFrame({
                    "Vade Dönemi": [f"{i}. Ay ({t.strftime('%B %Y')})" for i, t in enumerate(aylik_tarihler, 1)],
                    f"Öngörülen Seviye ({kod_karsi})": [f"{v:.4f}" for v in aylik_tahminler]
                })
                st.dataframe(tablo_ay, use_container_width=True)