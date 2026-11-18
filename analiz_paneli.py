import ccxt
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import yfinance as yf
import feedparser
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import json
import os
from datetime import datetime

st.set_page_config(page_title="Pro Kripto & Finans Analiz Terminali", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1.0rem; padding-bottom: 1rem; }
    div[data-testid="stHorizontalBlock"] button {
        font-weight: 600;
        border-radius: 6px;
    }
    /* ÜST SEKME KUTULARI (yalnızca sekme çubuğundaki butonlar; diğer butonlar etkilenmez) */
    .st-key-sekme_cubugu { margin-bottom: 0.7rem; }
    .st-key-sekme_cubugu button {
        min-height: 66px !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        border: 1px solid #30363d !important;
        background: #161b22 !important;
        color: #c9d1d9 !important;
        transition: all 0.15s ease;
    }
    .st-key-sekme_cubugu button p {
        font-size: 15px !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        white-space: normal !important;
    }
    .st-key-sekme_cubugu button:hover {
        border-color: #58a6ff !important;
        background: #1c2330 !important;
        color: #ffffff !important;
    }
    .st-key-sekme_cubugu button[kind="primary"],
    .st-key-sekme_cubugu button[data-testid="stBaseButton-primary"] {
        background: #1f6feb !important;
        border-color: #58a6ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(56, 139, 253, 0.35);
    }
    .stMetric {
        background-color: #161b22;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

IZLEME_DOSYASI = "izleme_listesi.json"

VARSAYILAN_LISTE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "TRXUSDT", "SHIBUSDT",
    "XNOUSDT", "RVNUSDT", "PIUSDT", "BTTCUSDT", "SUIUSDT",
    "DXY", "XAUUSD", "XAGUSD"
]

YAHOO_HARITASI = {
    "DXY": "DX-Y.NYB",
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "BIST100": "XU100.IS",
    "USDTRY": "TRY=X",
    "EURTRY": "EURTRY=X"
}

# Saniyelik periyotlar: Binance yalnızca 1 saniyelik (1s) mum sunar.
# 5s / 10s / 30s mumları 1 saniyelik mumların birleştirilmesiyle üretilir.
SANIYE_PERIYOTLARI = {"1s": 1, "5s": 5, "10s": 10, "30s": 30}
# Birleştirme için çekilecek en fazla 1 saniyelik mum sayısı (her 1000 mum = 1 istek)
SANIYE_HAM_MUM_LIMITI = 9000

def json_yukle(dosya, varsayilan):
    if os.path.exists(dosya):
        try:
            with open(dosya, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return varsayilan
    return varsayilan

def json_kaydet(dosya, veri):
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)

if "aktif_sekme" not in st.session_state:
    st.session_state.aktif_sekme = "analiz"
if "aktif_sembol" not in st.session_state:
    st.session_state.aktif_sembol = "BTCUSDT"
if "izleme_listesi" not in st.session_state:
    st.session_state.izleme_listesi = json_yukle(IZLEME_DOSYASI, VARSAYILAN_LISTE)

@st.cache_resource
def borsa_baglan():
    spot = ccxt.gate({"enableRateLimit": True})
    vadeli = ccxt.gate({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    return spot, vadeli

spot_borsa, vadeli_borsa = borsa_baglan()

def sembol_formatla(sembol):
    s = sembol.upper().strip()
    if s in YAHOO_HARITASI:
        return s, "yahoo"
    if "/" not in s and s.endswith("USDT"):
        return f"{s[:-4]}/USDT", "crypto"
    return s, "crypto"

@st.cache_data(ttl=15)
def mum_verisi_cek(sembol, zaman_dilimi, mum_limiti=1500):
    fmt_sembol, kaynak = sembol_formatla(sembol)
    if kaynak == "yahoo":
        yf_kod = YAHOO_HARITASI[fmt_sembol]
        yf_aralik_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk", "1M": "1mo",
            # Yahoo saniyelik veri sunmaz; en yakın çözünürlük olan 1 dakika kullanılır
            "1s": "1m", "5s": "1m", "10s": "1m", "30s": "1m"
        }
        aralik = yf_aralik_map.get(zaman_dilimi, "1d")
        periyot = "7d" if aralik in ["1m", "5m", "15m", "30m"] else ("2y" if aralik == "1h" else "max")
        df = yf.download(yf_kod, period=periyot, interval=aralik, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        tarih_kolon = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={
            tarih_kolon: "timestamp", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().tail(mum_limiti).reset_index(drop=True)

    ccxt_tf = zaman_dilimi
    saniye = SANIYE_PERIYOTLARI.get(zaman_dilimi)
    hedef_mum = mum_limiti
    if saniye:
        # Gerçek 1 saniyelik mumlar çekilir; 5s/10s/30s için aşağıda birleştirilir
        ccxt_tf = "1s"
        hedef_mum = min(mum_limiti * saniye, SANIYE_HAM_MUM_LIMITI)
    elif zaman_dilimi in ["3M", "6M", "1Y"]:
        ccxt_tf = "1M"

    tum_veriler = []
    since = None
    kalan = hedef_mum
    try:
        while kalan > 0:
            cekilecek = min(1000, kalan)
            if since is None:
                parca = spot_borsa.fetch_ohlcv(fmt_sembol, timeframe=ccxt_tf, limit=cekilecek)
            else:
                parca = spot_borsa.fetch_ohlcv(fmt_sembol, timeframe=ccxt_tf, limit=cekilecek, params={"endTime": since - 1})
            if not parca:
                break
            tum_veriler = parca + tum_veriler
            since = parca[0][0]
            kalan -= len(parca)
            if len(parca) < cekilecek:
                break
    except Exception:
        pass

    if not tum_veriler:
        return pd.DataFrame()

    df = pd.DataFrame(tum_veriler, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    if saniye and saniye > 1:
        # 1 saniyelik mumları N saniyelik mumlara birleştir (açılış=ilk, yüksek=max, düşük=min, kapanış=son, hacim=toplam)
        grup = df["timestamp"].dt.floor(f"{saniye}s")
        df = df.groupby(grup, sort=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), sayi=("close", "size")
        ).reset_index()
        # En eski kova eksik 1 saniyelik mumlarla kısmi kalabilir; hatalı OHLC göstermemek için at
        if len(df) > 1 and df["sayi"].iloc[0] < saniye:
            df = df.iloc[1:]
        df = df.drop(columns="sayi").tail(mum_limiti).reset_index(drop=True)
    return df

@st.cache_data(ttl=30)
def vadeli_veri_cek(sembol):
    fmt_sembol, kaynak = sembol_formatla(sembol)
    if kaynak != "crypto":
        return None, pd.DataFrame()
    try:
        fr_bilgi = vadeli_borsa.fetch_funding_rate(fmt_sembol)
        fr_oran = fr_bilgi.get("fundingRate", 0) * 100
    except Exception:
        fr_oran = None

    try:
        oi_veri = vadeli_borsa.fetch_open_interest_history(fmt_sembol, timeframe="1h", limit=100)
        oi_df = pd.DataFrame(oi_veri)
        if not oi_df.empty and "timestamp" in oi_df.columns:
            oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms")
    except Exception:
        oi_df = pd.DataFrame()

    return fr_oran, oi_df

def izleme_verisi_cek(liste):
    sonuclar = []
    kripto_semboller = []
    for s in liste:
        fmt, kaynak = sembol_formatla(s)
        if kaynak == "crypto":
            kripto_semboller.append((s, fmt))

    tickerlar = {}
    if kripto_semboller:
        try:
            tickerlar = spot_borsa.fetch_tickers()
        except Exception:
            tickerlar = {}

    for ham_s in liste:
        fmt, kaynak = sembol_formatla(ham_s)
        if kaynak == "crypto":
            t = tickerlar.get(fmt)
            if t:
                fiyat = t.get("last", 0) or 0
                degisim = t.get("percentage", 0) or 0
                sonuclar.append({"sembol": ham_s, "fiyat": fiyat, "degisim": degisim})
            else:
                sonuclar.append({"sembol": ham_s, "fiyat": 0.0, "degisim": 0.0})
        else:
            try:
                yf_kod = YAHOO_HARITASI[fmt]
                hist = yf.Ticker(yf_kod).history(period="5d")
                if len(hist) >= 2:
                    son = float(hist["Close"].iloc[-1])
                    onceki = float(hist["Close"].iloc[-2])
                    deg = ((son - onceki) / onceki) * 100
                    sonuclar.append({"sembol": ham_s, "fiyat": son, "degisim": deg})
                elif len(hist) == 1:
                    sonuclar.append({"sembol": ham_s, "fiyat": float(hist["Close"].iloc[-1]), "degisim": 0.0})
                else:
                    sonuclar.append({"sembol": ham_s, "fiyat": 0.0, "degisim": 0.0})
            except Exception:
                sonuclar.append({"sembol": ham_s, "fiyat": 0.0, "degisim": 0.0})
    return sonuclar

# ÜST GÖRÜNÜR SEKME ÇUBUĞU
SEKME_TANIMLARI = [
    ("analiz", "📈 Canlı Teknik Analiz Terminali"),
    ("emir_defteri", "📖 Canlı Emir Defteri & Alıcı/Satıcı Baskısı"),
    ("haberler", "🌍 BIST / Forex & Küresel Merkez Bankaları"),
]
# key="sekme_cubugu" sayesinde yukarıdaki CSS yalnızca bu butonlara uygulanır
with st.container(key="sekme_cubugu"):
    sekme_kolonlari = st.columns([1, 1, 1, 0.8])  # son kolon boş: kutular ekranı baştan sona kaplamasın
    for sekme_kolon, (sekme_id, sekme_baslik) in zip(sekme_kolonlari, SEKME_TANIMLARI):
        with sekme_kolon:
            if st.button(sekme_baslik, key=f"sekme_{sekme_id}", use_container_width=True,
                         type="primary" if st.session_state.aktif_sekme == sekme_id else "secondary"):
                st.session_state.aktif_sekme = sekme_id
                st.rerun()

# SOL KONTROL PANELİ (SIDEBAR)
with st.sidebar:
    st.header("⚙️ Terminal Ayarları")
    canli_liste = st.checkbox("⚡ İzleme Listesi Canlı Akış (2 sn)", value=True)

    girilen_sembol = st.text_input("Parite Ara (Örn: BTCUSDT, XAUUSD)", value=st.session_state.aktif_sembol).upper().strip()
    if girilen_sembol and girilen_sembol != st.session_state.aktif_sembol:
        st.session_state.aktif_sembol = girilen_sembol
        st.rerun()

    zaman_dilimleri = [
        "1s", "5s", "10s", "30s", "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M", "3M", "6M", "1Y"
    ]
    zaman_dilimi = st.selectbox("⏱️ Zaman Dilimi", zaman_dilimleri, index=zaman_dilimleri.index("15m"))

    st.subheader("📈 Ana Grafik İndikatörleri")
    goster_ema = st.checkbox("EMA (20, 50, 200)", value=True)
    goster_sma = st.checkbox("SMA (20, 50, 200)", value=False)
    goster_bb = st.checkbox("Bollinger Bantları (20, 2)", value=True)
    goster_fib = st.checkbox("Otomatik Fibonacci Seviyeleri", value=False)
    goster_rvp = st.checkbox("RVP / Hacim Profili (POC)", value=True)
    goster_hacim = st.checkbox("Hacim (Volume)", value=True)

    st.subheader("📉 Alt Osilatörler & Vadeli")
    goster_rsi = st.checkbox("RSI (14)", value=True)
    goster_macd = st.checkbox("MACD (12, 26, 9)", value=True)
    goster_stok = st.checkbox("Stokastik Osilatör (80/20)", value=False)
    goster_cci = st.checkbox("CCI (Emtia Kanal Endeksi)", value=False)
    goster_obv = st.checkbox("OBV (Denge İşlem Hacmi)", value=False)
    goster_oi = st.checkbox("Açık Pozisyon (Open Interest)", value=False)

# ==============================================================================
# SEKME 1: CANLI TEKNİK ANALİZ TERMİNALİ + İZLEME LİSTESİ
# ==============================================================================
if st.session_state.aktif_sekme == "analiz":
    ana_col, liste_col = st.columns([3.3, 1.1])

    with ana_col:
        df = mum_verisi_cek(st.session_state.aktif_sembol, zaman_dilimi, 1500)
        if df.empty:
            st.error(f"'{st.session_state.aktif_sembol}' için veri bulunamadı. Parite adını kontrol edin.")
        else:
            # Grafik zamanları artan ve tekil olmak zorunda; veri kaynağından gelen olası tekrarları temizle
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            df["RSI"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)

            for p in [20, 50, 200]:
                df[f"EMA{p}"] = ta.ema(df["close"], length=p)
                df[f"SMA{p}"] = ta.sma(df["close"], length=p)

            bb = ta.bbands(df["close"], length=20, std=2)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)

            stok = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
            if stok is not None:
                df = pd.concat([df, stok], axis=1)

            df["CCI"] = ta.cci(df["high"], df["low"], df["close"], length=20)
            df["OBV"] = ta.obv(df["close"], df["volume"])

            fr_oran, oi_df = vadeli_veri_cek(st.session_state.aktif_sembol)

            son_fiyat = float(df["close"].iloc[-1])
            son_rsi = float(df["RSI"].iloc[-1]) if pd.notna(df["RSI"].iloc[-1]) else 0.0
            son_zaman = df["timestamp"].iloc[-1].strftime("%d.%m %H:%M")

            # Üst Bilgi Kartları (Canlı Fiyat + RSI + Fonlama + OI + Son Mum)
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                @st.fragment(run_every=2 if canli_liste else None)
                def canli_ust_fiyat_karti():
                    fmt_s, kynk = sembol_formatla(st.session_state.aktif_sembol)
                    guncel_f = son_fiyat
                    deg_yuzde = 0.0
                    if kynk == "crypto":
                        try:
                            tk = spot_borsa.fetch_ticker(fmt_s)
                            guncel_f = float(tk.get("last") or son_fiyat)
                            deg_yuzde = float(tk.get("percentage") or 0.0)
                        except Exception:
                            pass
                    f_str = f"{guncel_f:,.4f}" if guncel_f < 10 else f"{guncel_f:,.2f}"
                    st.metric(f"🟢 {st.session_state.aktif_sembol} Canlı", f_str, f"%{deg_yuzde:+.2f}")
                canli_ust_fiyat_karti()

            m2.metric("RSI (14)", f"{son_rsi:.2f}")
            m3.metric("Fonlama Oranı", f"%{fr_oran:+.4f}" if fr_oran is not None else "Spot / Forex")
            m4.metric("Açık Pozisyon (OI)", f"{oi_df['openInterestAmount'].iloc[-1]:,.0f}" if not oi_df.empty and "openInterestAmount" in oi_df.columns else "-")
            m5.metric("Son Mum Zamanı", son_zaman)

            # JSON Veri Hazırlığı (Lightweight Charts İçin - Yerel Saat Dilimi Uyumlu)
            # Türkiye saati (+3 saat) ile grafik ekseninde tam yerel saati göstermek için +10800 saniye ekliyoruz
            # NOT: pandas 3.x'te datetime64 çözünürlüğü (ms/us/ns) değişebildiği için astype("int64") // 10**9
            # yanlış saniye değeri üretir (tüm mumlar aynı zamanı alır ve grafik hata verir).
            # Çözünürlükten bağımsız olan Timedelta bölmesi kullanılır.
            ts_unix = ((df["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")) + 10800

            candles_json = json.dumps([
                {"time": int(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
                for t, o, h, l, c in zip(ts_unix, df["open"], df["high"], df["low"], df["close"])
            ])
            volumes_json = json.dumps([
                {"time": int(t), "value": float(v), "color": "rgba(38, 166, 154, 0.55)" if c >= o else "rgba(239, 83, 80, 0.55)"}
                for t, v, o, c in zip(ts_unix, df["volume"], df["open"], df["close"])
            ]) if goster_hacim else "[]"

            def seri_json(kolon):
                if kolon not in df.columns:
                    return "[]"
                return json.dumps([
                    {"time": int(t), "value": float(v)}
                    for t, v in zip(ts_unix, df[kolon]) if pd.notna(v)
                ])

            ema20_j = seri_json("EMA20") if goster_ema else "[]"
            ema50_j = seri_json("EMA50") if goster_ema else "[]"
            ema200_j = seri_json("EMA200") if goster_ema else "[]"
            sma20_j = seri_json("SMA20") if goster_sma else "[]"
            sma50_j = seri_json("SMA50") if goster_sma else "[]"
            sma200_j = seri_json("SMA200") if goster_sma else "[]"

            bbu_kol = [c for c in df.columns if c.startswith("BBU_")]
            bbl_kol = [c for c in df.columns if c.startswith("BBL_")]
            bbu_j = seri_json(bbu_kol[0]) if (goster_bb and bbu_kol) else "[]"
            bbl_j = seri_json(bbl_kol[0]) if (goster_bb and bbl_kol) else "[]"

            fib_list = []
            if goster_fib:
                gorunen = df.tail(160)
                en_y, en_d = float(gorunen["high"].max()), float(gorunen["low"].min())
                fark = en_y - en_d
                for sev in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]:
                    fib_list.append({"price": en_y - fark * sev, "label": f"Fib {sev}"})
            fib_json = json.dumps(fib_list)

            poc_price = None
            if goster_rvp and df["volume"].sum() > 0:
                gorunen = df.tail(160)
                hist_v, bin_edges = np.histogram(gorunen["close"], bins=24, weights=gorunen["volume"])
                poc_idx = int(np.argmax(hist_v))
                poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)
            poc_json = json.dumps(poc_price)

            fmt_s, kaynak = sembol_formatla(st.session_state.aktif_sembol)
            ws_symbol = fmt_s.replace("/", "").lower() if kaynak == "crypto" else ""
            # Saniyelik periyotlarda canlı akış kline_1s'dir; 5s/10s/30s için mumlar tarayıcıda birleştirilir
            tf_map = {"1s": "1s", "5s": "1s", "10s": "1s", "30s": "1s", "3M": "1M", "6M": "1M", "1Y": "1M"}
            ws_tf = tf_map.get(zaman_dilimi, zaman_dilimi)
            agg_n = SANIYE_PERIYOTLARI.get(zaman_dilimi, 1)
            seconds_gorunur = "true" if zaman_dilimi in SANIYE_PERIYOTLARI else "false"
            if kaynak == "yahoo" and zaman_dilimi in SANIYE_PERIYOTLARI:
                st.info("Saniyelik veri yalnızca kripto pariteleri için mevcut; bu sembolde 1 dakikalık mumlar gösteriliyor.")
            storage_key = f"tv_drawings_{st.session_state.aktif_sembol}_{zaman_dilimi}"

            # ALT OSİLATÖR & VADELİ PANELLERİ
            # Ana grafikle AYNI iframe içinde çizilir; böylece yakınlaşma / uzaklaşma / kaydırma hepsinde senkron olur.
            def osc_seri(kolon, renk, kalinlik=1.5):
                return {"type": "line", "color": renk, "width": kalinlik, "data": json.loads(seri_json(kolon))}

            osc_defs = []
            if goster_rsi and "RSI" in df.columns:
                osc_defs.append({
                    "title": "RSI (14)", "height": 160, "range": [0, 100],
                    "levels": [{"price": 70, "color": "rgba(239, 83, 80, 0.8)"}, {"price": 30, "color": "rgba(38, 166, 154, 0.8)"}],
                    "series": [osc_seri("RSI", "#ba68c8", 1.6)]
                })
            if goster_macd and "MACD_12_26_9" in df.columns:
                macd_hist = [
                    {"time": int(t), "value": float(v), "color": "#26a69a" if v >= 0 else "#ef5350"}
                    for t, v in zip(ts_unix, df["MACDh_12_26_9"]) if pd.notna(v)
                ]
                osc_defs.append({
                    "title": "MACD (12, 26, 9)", "height": 170,
                    "levels": [{"price": 0, "color": "rgba(255, 255, 255, 0.35)"}],
                    "series": [
                        {"type": "hist", "data": macd_hist},
                        osc_seri("MACD_12_26_9", "#29b6f6"),
                        osc_seri("MACDs_12_26_9", "#ffa726"),
                    ]
                })
            if goster_stok and "STOCHk_14_3_3" in df.columns:
                osc_defs.append({
                    "title": "Stokastik Osilatör (80 / 20)", "height": 160, "range": [0, 100],
                    "levels": [{"price": 80, "color": "rgba(239, 83, 80, 0.8)"}, {"price": 20, "color": "rgba(38, 166, 154, 0.8)"}],
                    "series": [osc_seri("STOCHk_14_3_3", "#29b6f6"), osc_seri("STOCHd_14_3_3", "#ff7043")]
                })
            if goster_cci and "CCI" in df.columns:
                osc_defs.append({
                    "title": "CCI (Emtia Kanal Endeksi)", "height": 160,
                    "levels": [{"price": 100, "color": "rgba(239, 83, 80, 0.8)"}, {"price": -100, "color": "rgba(38, 166, 154, 0.8)"}],
                    "series": [osc_seri("CCI", "#ffd54f")]
                })
            if goster_obv and "OBV" in df.columns:
                osc_defs.append({
                    "title": "OBV (Denge İşlem Hacmi)", "height": 160,
                    "series": [osc_seri("OBV", "#4db6ac")]
                })
            if goster_oi and oi_df is not None and not oi_df.empty and {"timestamp", "openInterestAmount"} <= set(oi_df.columns):
                oi_tmp = oi_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                oi_ts = ((oi_tmp["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")) + 10800
                oi_data = [
                    {"time": int(t), "value": float(v), "color": "#7986cb"}
                    for t, v in zip(oi_ts, oi_tmp["openInterestAmount"]) if pd.notna(v)
                ]
                # OI verisi saatlik ve ayrı bir zaman ekseninde olduğu için zaman bazlı senkronize edilir
                osc_defs.append({
                    "title": "Açık Pozisyon (Open Interest)", "height": 160, "sync": "time",
                    "series": [{"type": "hist", "data": oi_data}]
                })

            osc_json = json.dumps(osc_defs)
            iframe_yuksekligi = 645 + sum(d["height"] + 2 for d in osc_defs)

            # TRADINGVIEW LIGHTWEIGHT CHARTS + KESİNTİSİZ ÇİZİM + HACİM ALTI TARİH ŞERİDİ
            tv_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
            <script>
                // unpkg erişilemezse yedek CDN'den yükle
                if (typeof LightweightCharts === 'undefined') {{
                    document.write('<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"><\\/script>');
                }}
                // Grafikte bir hata olursa sessizce boş kalmak yerine ekranın altında göster
                window.addEventListener('error', function (e) {{
                    var d = document.getElementById('err');
                    if (!d) {{ d = document.createElement('div'); d.id = 'err'; document.body.appendChild(d); }}
                    d.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99;background:#3d1c1c;color:#ff7b72;font:12px monospace;padding:6px 10px;';
                    d.textContent = 'Grafik hatası: ' + e.message;
                }});
            </script>
            <style>
                * {{ box-sizing: border-box; }}
                body {{ margin: 0; padding: 0; background: #0e1117; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; overflow: hidden; }}
                #toolbar {{
                    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
                    background: #161b22; padding: 7px 10px; border: 1px solid #30363d; border-radius: 6px 6px 0 0;
                    height: 44px;
                }}
                .tool-btn {{
                    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                    padding: 4px 9px; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: 600;
                }}
                .tool-btn:hover {{ background: #30363d; color: #ffffff; }}
                .tool-btn.active {{ background: #1f6feb; color: #ffffff; border-color: #388bfd; }}
                .tool-btn.danger {{ background: #3d1c1c; border-color: #f85149; color: #ff7b72; }}
                #live-badge {{
                    margin-left: auto; font-size: 13px; font-weight: 700; color: #26a69a;
                    background: #0d1117; padding: 4px 10px; border-radius: 5px; border: 1px solid #26a69a;
                }}
                #chart-wrap {{ position: relative; width: 100%; height: 590px; border: 1px solid #30363d; border-top: none; background: #0e1117; }}
                #chart-container {{ width: 100%; height: 100%; }}
                #draw-canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10; }}
                .osc-box {{ position: relative; width: 100%; border: 1px solid #30363d; border-top: none; background: #0e1117; }}
                .osc-title {{ position: absolute; top: 4px; left: 8px; z-index: 5; font-size: 11px; font-weight: 600; color: #8b949e; pointer-events: none; }}
            </style>
            </head>
            <body>
            <div id="toolbar">
                <button class="tool-btn active" id="btn-pan" onclick="setMode('pan')">✋ Gezin / Kaydır</button>
                <button class="tool-btn" id="btn-line" onclick="setMode('line')">📏 Trend Çizgisi</button>
                <button class="tool-btn" id="btn-ray" onclick="setMode('ray')">↗️ Işın</button>
                <button class="tool-btn" id="btn-hline" onclick="setMode('hline')">➖ Yatay Çizgi</button>
                <button class="tool-btn" id="btn-rect" onclick="setMode('rect')">⬜ Kanal / Kutu</button>
                <button class="tool-btn" id="btn-log" onclick="toggleLog()">LOG</button>
                <button class="tool-btn" onclick="resetChart()">🔄 Sıfırla</button>
                <button class="tool-btn" onclick="undoDraw()">↩️ Geri Al</button>
                <button class="tool-btn danger" onclick="clearDraws()">🗑️ Tümünü Sil</button>
                <div id="live-badge">🟢 {st.session_state.aktif_sembol}: {son_fiyat:.4f}</div>
            </div>
            <div id="chart-wrap">
                <div id="chart-container"></div>
                <canvas id="draw-canvas"></canvas>
            </div>
            <div id="osc-wrap"></div>

            <script>
            const chartContainer = document.getElementById('chart-container');
            const canvas = document.getElementById('draw-canvas');
            const ctx = canvas.getContext('2d');

            const chart = LightweightCharts.createChart(chartContainer, {{
                width: chartContainer.clientWidth,
                height: 590,
                layout: {{
                    background: {{ type: 'solid', color: '#0e1117' }},
                    textColor: '#d1d4dc',
                    fontSize: 12
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(42, 46, 57, 0.45)' }},
                    horzLines: {{ color: 'rgba(42, 46, 57, 0.45)' }}
                }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                rightPriceScale: {{
                    borderColor: '#30363d',
                    scaleMargins: {{ top: 0.06, bottom: 0.24 }}
                }},
                timeScale: {{
                    borderColor: '#30363d',
                    visible: true,
                    timeVisible: true,
                    secondsVisible: {seconds_gorunur},
                    rightOffset: 6,
                    barSpacing: 8,
                    minBarSpacing: 1.5
                }}
            }});

            const candleSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a', downColor: '#ef5350',
                borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350'
            }});
            const candlesData = {candles_json};
            candleSeries.setData(candlesData);

            const volumeData = {volumes_json};
            if (volumeData.length > 0) {{
                const volumeSeries = chart.addHistogramSeries({{
                    priceFormat: {{ type: 'volume' }},
                    priceScaleId: 'vol'
                }});
                chart.priceScale('vol').applyOptions({{
                    scaleMargins: {{ top: 0.78, bottom: 0.02 }}
                }});
                volumeSeries.setData(volumeData);
            }}

            function addLine(data, color, width, style=0) {{
                if (!data || data.length === 0) return;
                const s = chart.addLineSeries({{ color: color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false }});
                s.setData(data);
            }}
            addLine({ema20_j}, '#ff9800', 1.5);
            addLine({ema50_j}, '#2196f3', 1.5);
            addLine({ema200_j}, '#e91e63', 2);
            addLine({sma20_j}, '#8bc34a', 1.5, 2);
            addLine({sma50_j}, '#9c27b0', 1.5, 2);
            addLine({sma200_j}, '#ffeb3b', 2, 2);
            addLine({bbu_j}, 'rgba(0, 188, 212, 0.75)', 1.2);
            addLine({bbl_j}, 'rgba(0, 188, 212, 0.75)', 1.2);

            const fibLevels = {fib_json};
            fibLevels.forEach(f => {{
                candleSeries.createPriceLine({{
                    price: f.price, color: 'rgba(255, 255, 255, 0.35)', lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: f.label
                }});
            }});

            const pocPrice = {poc_json};
            if (pocPrice !== null) {{
                candleSeries.createPriceLine({{
                    price: pocPrice, color: '#ff5722', lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: true, title: 'RVP POC'
                }});
            }}

            // Başlangıçta son 130 muma odaklan
            if (candlesData.length > 130) {{
                chart.timeScale().setVisibleLogicalRange({{
                    from: candlesData.length - 130,
                    to: candlesData.length + 5
                }});
            }}

            // Saniyelik Canlı Binance WebSocket Akışı
            const wsSymbol = "{ws_symbol}";
            const wsTf = "{ws_tf}";
            // 5s / 10s / 30s: gelen 1 saniyelik mumları N saniyelik kovalara birleştir (1s ve diğer periyotlarda AGG_N = 1)
            const AGG_N = {agg_n};
            let curBar = candlesData.length ? Object.assign({{}}, candlesData[candlesData.length - 1]) : null;
            if (wsSymbol !== "") {{
                const ws = new WebSocket(`wss://stream.binance.com:9443/ws/${{wsSymbol}}@kline_${{wsTf}}`);
                ws.onmessage = (event) => {{
                    const msg = JSON.parse(event.data);
                    if (msg.k) {{
                        const k = msg.k;
                        const bar = {{
                            time: Math.floor(k.t / 1000 / AGG_N) * AGG_N + 10800,
                            open: parseFloat(k.o),
                            high: parseFloat(k.h),
                            low: parseFloat(k.l),
                            close: parseFloat(k.c)
                        }};
                        if (AGG_N > 1) {{
                            if (curBar && curBar.time === bar.time) {{
                                // Aynı kova: açılışı koru, yüksek/düşük/kapanışı güncelle
                                bar.open = curBar.open;
                                bar.high = Math.max(curBar.high, bar.high);
                                bar.low = Math.min(curBar.low, bar.low);
                            }} else if (curBar && bar.time < curBar.time) {{
                                return;
                            }}
                            curBar = Object.assign({{}}, bar);
                        }}
                        candleSeries.update(bar);
                        // Alt panellerin zaman ekseni de yeni mumla birlikte uzasın (senkron kalması için)
                        try {{ oscCharts.forEach(o => {{ if (o.ws) o.ws.update({{ time: bar.time }}); }}); }} catch (err) {{}}
                        const priceFmt = bar.close < 10 ? bar.close.toFixed(4) : bar.close.toFixed(2);
                        document.getElementById('live-badge').innerText = `🟢 {st.session_state.aktif_sembol}: ${{priceFmt}}`;
                    }}
                }};
            }}

            // ÇİZİM ARAÇLARI MOTORU
            let drawMode = 'pan';
            let isLog = false;
            let drawings = JSON.parse(localStorage.getItem("{storage_key}") || "[]");
            let currentDraw = null;

            function resizeCanvas() {{
                canvas.width = chartContainer.clientWidth;
                canvas.height = chartContainer.clientHeight;
                chart.applyOptions({{ width: chartContainer.clientWidth }});
                renderDrawings();
            }}
            window.addEventListener('resize', resizeCanvas);
            resizeCanvas();

            function setMode(m) {{
                drawMode = m;
                document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
                const btn = document.getElementById('btn-' + m);
                if (btn) btn.classList.add('active');
                canvas.style.pointerEvents = (m === 'pan') ? 'none' : 'auto';
            }}

            function toggleLog() {{
                isLog = !isLog;
                chart.priceScale('right').applyOptions({{ mode: isLog ? LightweightCharts.PriceScaleMode.Logarithmic : LightweightCharts.PriceScaleMode.Normal }});
                document.getElementById('btn-log').classList.toggle('active', isLog);
            }}

            function resetChart() {{
                setMode('pan');
                if (candlesData.length > 130) {{
                    chart.timeScale().setVisibleLogicalRange({{ from: candlesData.length - 130, to: candlesData.length + 5 }});
                }} else {{
                    chart.timeScale().fitContent();
                }}
            }}

            function undoDraw() {{
                drawings.pop();
                localStorage.setItem("{storage_key}", JSON.stringify(drawings));
                renderDrawings();
            }}

            function clearDraws() {{
                drawings = [];
                localStorage.setItem("{storage_key}", JSON.stringify(drawings));
                renderDrawings();
            }}

            function coordToData(x, y) {{
                const logical = chart.timeScale().coordinateToLogical(x);
                const price = candleSeries.coordinateToPrice(y);
                return {{ logical, price }};
            }}

            function dataToCoord(logical, price) {{
                const x = chart.timeScale().logicalToCoordinate(logical);
                const y = candleSeries.priceToCoordinate(price);
                return {{ x, y }};
            }}

            canvas.addEventListener('mousedown', (e) => {{
                if (drawMode === 'pan') return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const p1 = coordToData(x, y);
                if (p1.logical === null || p1.price === null) return;
                if (drawMode === 'hline') {{
                    drawings.push({{ type: 'hline', p1: p1, p2: p1 }});
                    localStorage.setItem("{storage_key}", JSON.stringify(drawings));
                    renderDrawings();
                    setMode('pan');
                    return;
                }}
                currentDraw = {{ type: drawMode, p1: p1, p2: p1 }};
            }});

            canvas.addEventListener('mousemove', (e) => {{
                if (!currentDraw) return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const p2 = coordToData(x, y);
                if (p2.logical !== null && p2.price !== null) {{
                    currentDraw.p2 = p2;
                    renderDrawings();
                }}
            }});

            canvas.addEventListener('mouseup', () => {{
                if (currentDraw) {{
                    drawings.push(currentDraw);
                    localStorage.setItem("{storage_key}", JSON.stringify(drawings));
                    currentDraw = null;
                    renderDrawings();
                    setMode('pan');
                }}
            }});

            function renderDrawings() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const all = currentDraw ? [...drawings, currentDraw] : drawings;
                all.forEach(d => {{
                    const c1 = dataToCoord(d.p1.logical, d.p1.price);
                    const c2 = dataToCoord(d.p2.logical, d.p2.price);
                    if (c1.y === null) return;
                    ctx.strokeStyle = '#00e5ff';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    if (d.type === 'hline') {{
                        ctx.moveTo(0, c1.y);
                        ctx.lineTo(canvas.width, c1.y);
                        ctx.stroke();
                    }} else if (d.type === 'line' && c1.x !== null && c2.x !== null && c2.y !== null) {{
                        ctx.moveTo(c1.x, c1.y);
                        ctx.lineTo(c2.x, c2.y);
                        ctx.stroke();
                    }} else if (d.type === 'ray' && c1.x !== null && c2.x !== null && c2.y !== null) {{
                        const dx = c2.x - c1.x;
                        const dy = c2.y - c1.y;
                        ctx.moveTo(c1.x, c1.y);
                        ctx.lineTo(c1.x + dx * 50, c1.y + dy * 50);
                        ctx.stroke();
                    }} else if (d.type === 'rect' && c1.x !== null && c2.x !== null && c2.y !== null) {{
                        ctx.fillStyle = 'rgba(0, 229, 255, 0.12)';
                        ctx.fillRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
                        ctx.strokeRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
                    }}
                }});
            }}

            chart.timeScale().subscribeVisibleLogicalRangeChange(renderDrawings);

            // ==========================================================================
            // ALT OSİLATÖR / VADELİ PANELLERİ — ana grafikle senkron zoom & kaydırma
            // ==========================================================================
            const oscDefs = {osc_json};
            const oscCharts = [];
            const oscWrap = document.getElementById('osc-wrap');
            // Tüm panellerin zaman ekseni ana grafikle birebir aynı olsun diye her panele mum zamanlarından
            // oluşan görünmez (whitespace) bir seri eklenir; indikatörlerin ısınma boşlukları ekseni kaydırmaz.
            const baseTimes = candlesData.map(d => ({{ time: d.time }}));

            oscDefs.forEach(def => {{
                const isTime = def.sync === 'time';
                const box = document.createElement('div');
                box.className = 'osc-box';
                box.style.height = def.height + 'px';
                const holder = document.createElement('div');
                holder.style.cssText = 'width:100%;height:100%;';
                const label = document.createElement('div');
                label.className = 'osc-title';
                label.textContent = def.title;
                box.appendChild(holder);
                box.appendChild(label);
                oscWrap.appendChild(box);

                const c = LightweightCharts.createChart(holder, {{
                    width: holder.clientWidth,
                    height: def.height,
                    layout: {{ background: {{ type: 'solid', color: '#0e1117' }}, textColor: '#d1d4dc', fontSize: 11 }},
                    grid: {{
                        vertLines: {{ color: 'rgba(42, 46, 57, 0.45)' }},
                        horzLines: {{ color: 'rgba(42, 46, 57, 0.45)' }}
                    }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    rightPriceScale: {{ borderColor: '#30363d', minimumWidth: 78, scaleMargins: {{ top: 0.16, bottom: 0.1 }} }},
                    // Tarih etiketi yalnızca ana grafikte (hacmin hemen altında) görünür
                    timeScale: {{ visible: false, borderColor: '#30363d', rightOffset: 6, barSpacing: 8, minBarSpacing: 1.5 }},
                    handleScroll: !isTime,
                    handleScale: !isTime
                }});

                let wsSeries = null;
                if (!isTime) {{
                    wsSeries = c.addLineSeries({{ color: 'rgba(0,0,0,0)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }});
                    wsSeries.setData(baseTimes);
                }}

                let first = null;
                def.series.forEach(s => {{
                    let ser;
                    if (s.type === 'hist') {{
                        ser = c.addHistogramSeries({{ priceLineVisible: false, lastValueVisible: false, color: '#7986cb' }});
                    }} else {{
                        ser = c.addLineSeries({{ color: s.color, lineWidth: s.width || 1.5, priceLineVisible: false, lastValueVisible: false }});
                    }}
                    ser.setData(s.data);
                    if (!first) first = ser;
                }});

                if (first && def.range) {{
                    first.applyOptions({{
                        autoscaleInfoProvider: () => ({{ priceRange: {{ minValue: def.range[0], maxValue: def.range[1] }} }})
                    }});
                }}
                if (first && def.levels) {{
                    def.levels.forEach(l => first.createPriceLine({{
                        price: l.price, color: l.color, lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: false, title: ''
                    }}));
                }}
                oscCharts.push({{ chart: c, holder: holder, isTime: isTime, ws: wsSeries }});
            }});

            // --- Senkronizasyon: hangi grafikte yakınlaşılırsa / kaydırılırsa diğerleri aynı aralığa gelir ---
            let syncing = false;
            function idxClamp(v) {{
                return Math.max(0, Math.min(candlesData.length - 1, Math.round(v)));
            }}
            function applySync(srcChart, range) {{
                if (syncing || !range) return;
                syncing = true;
                try {{
                    if (srcChart !== chart) chart.timeScale().setVisibleLogicalRange(range);
                    oscCharts.forEach(o => {{
                        if (o.chart === srcChart) return;
                        if (!o.isTime) {{
                            o.chart.timeScale().setVisibleLogicalRange(range);
                        }} else if (candlesData.length > 0) {{
                            // OI saatlik/ayrı eksende: mantıksal aralığı mum zamanına çevirip zaman aralığı olarak uygula
                            try {{
                                o.chart.timeScale().setVisibleRange({{
                                    from: candlesData[idxClamp(range.from)].time,
                                    to: candlesData[idxClamp(range.to)].time
                                }});
                            }} catch (err) {{}}
                        }}
                    }});
                }} finally {{
                    syncing = false;
                }}
            }}
            chart.timeScale().subscribeVisibleLogicalRangeChange(r => applySync(chart, r));
            oscCharts.forEach(o => {{
                if (!o.isTime) o.chart.timeScale().subscribeVisibleLogicalRangeChange(r => applySync(o.chart, r));
            }});

            // İlk açılışta tüm paneller ana grafiğin görünür aralığıyla başlasın
            function ilkSenkron() {{
                const r = chart.timeScale().getVisibleLogicalRange();
                if (r) applySync(chart, r);
            }}
            ilkSenkron();
            setTimeout(ilkSenkron, 60);

            // Fiyat ekseni genişliklerini eşitle: aksi halde mumlar paneller arasında dikey olarak hizalanmaz
            function alignScales() {{
                try {{
                    let w = 78;
                    w = Math.max(w, chart.priceScale('right').width());
                    oscCharts.forEach(o => {{ w = Math.max(w, o.chart.priceScale('right').width()); }});
                    chart.priceScale('right').applyOptions({{ minimumWidth: w }});
                    oscCharts.forEach(o => o.chart.priceScale('right').applyOptions({{ minimumWidth: w }}));
                }} catch (err) {{}}
            }}
            setTimeout(alignScales, 150);
            setTimeout(alignScales, 1200);

            window.addEventListener('resize', () => {{
                oscCharts.forEach(o => o.chart.applyOptions({{ width: o.holder.clientWidth }}));
            }});
            </script>
            </body>
            </html>
            """
            components.html(tv_html, height=iframe_yuksekligi, scrolling=False)

    # SAĞ İZLEME LİSTESİ (100 KAPASİTE + EKLEME / SİLME KUTUSU)
    with liste_col:
        st.subheader(f"📋 İzleme Listesi ({len(st.session_state.izleme_listesi)}/100)")

        ekle_c1, ekle_c2 = st.columns([2.2, 1])
        with ekle_c1:
            yeni_coin = st.text_input("Parite Ekle", placeholder="Örn: SOLUSDT", label_visibility="collapsed").upper().strip()
        with ekle_c2:
            if st.button("➕ Ekle", use_container_width=True):
                if yeni_coin and yeni_coin not in st.session_state.izleme_listesi:
                    if len(st.session_state.izleme_listesi) < 100:
                        st.session_state.izleme_listesi.append(yeni_coin)
                        json_kaydet(IZLEME_DOSYASI, st.session_state.izleme_listesi)
                        st.rerun()

        silinecek_secim = st.selectbox("Listeden Çıkar (Silme Kutusu)", ["Seçiniz..."] + st.session_state.izleme_listesi, label_visibility="collapsed")
        if silinecek_secim != "Seçiniz...":
            if st.button(f"🗑️ '{silinecek_secim}' Listeden Sil", use_container_width=True):
                st.session_state.izleme_listesi.remove(silinecek_secim)
                json_kaydet(IZLEME_DOSYASI, st.session_state.izleme_listesi)
                st.rerun()

        @st.fragment(run_every=2 if canli_liste else None)
        def canli_izleme_listesi():
            veriler = izleme_verisi_cek(st.session_state.izleme_listesi)
            for item in veriler:
                s = item["sembol"]
                f = item["fiyat"]
                d = item["degisim"]
                renk = "#26a69a" if d >= 0 else "#ef5350"
                isaret = "+" if d >= 0 else ""
                fiyat_str = f"{f:,.4f}" if f < 10 else f"{f:,.2f}"

                ic1, ic2, ic3 = st.columns([1.8, 1.8, 0.7])
                with ic1:
                    if st.button(s, key=f"sec_{s}", use_container_width=True):
                        st.session_state.aktif_sembol = s
                        st.rerun()
                with ic2:
                    st.markdown(f"<div style='text-align:right; font-size:13px; line-height:1.2;'><b>{fiyat_str}</b><br><span style='color:{renk};'>{isaret}{d:.2f}%</span></div>", unsafe_allow_html=True)
                with ic3:
                    if st.button("❌", key=f"sil_{s}", help=f"{s} sil"):
                        st.session_state.izleme_listesi.remove(s)
                        json_kaydet(IZLEME_DOSYASI, st.session_state.izleme_listesi)
                        st.rerun()

        canli_izleme_listesi()

# ==============================================================================
# SEKME 2: CANLI EMİR DEFTERİ & ALICI / SATICI BASKISI (2 SN CANLI AKIŞ)
# ==============================================================================
elif st.session_state.aktif_sekme == "emir_defteri":
    st.subheader(f"📖 {st.session_state.aktif_sembol} Canlı Emir Defteri Derinliği & Alıcı/Satıcı Baskısı")

    @st.fragment(run_every=2)
    def canli_emir_defteri_paneli():
        fmt_sembol, kaynak = sembol_formatla(st.session_state.aktif_sembol)
        if kaynak != "crypto":
            st.warning("Emir defteri derinlik verisi yalnızca kripto pariteleri (Örn: BTCUSDT, ETHUSDT) için canlı olarak çekilebilmektedir.")
            return

        try:
            ob = spot_borsa.fetch_order_book(fmt_sembol, limit=100)
            alislar = pd.DataFrame(ob["bids"], columns=["fiyat", "miktar"])
            satislar = pd.DataFrame(ob["asks"], columns=["fiyat", "miktar"])

            alislar["toplam_usdt"] = alislar["fiyat"] * alislar["miktar"]
            satislar["toplam_usdt"] = satislar["fiyat"] * satislar["miktar"]

            toplam_alis_hacim = alislar["toplam_usdt"].sum()
            toplam_satis_hacim = satislar["toplam_usdt"].sum()
            genel_toplam = toplam_alis_hacim + toplam_satis_hacim

            alici_yuzde = (toplam_alis_hacim / genel_toplam * 100) if genel_toplam > 0 else 50.0
            satici_yuzde = 100.0 - alici_yuzde

            b1, b2, b3 = st.columns(3)
            b1.metric("🟢 Alıcı Baskısı (Bid)", f"%{alici_yuzde:.2f}", f"{toplam_alis_hacim:,.0f} USDT")
            b2.metric("🔴 Satıcı Baskısı (Ask)", f"%{satici_yuzde:.2f}", f"{toplam_satis_hacim:,.0f} USDT")
            durum_metni = "GÜÇLÜ ALICI BASKISI" if alici_yuzde > 55 else ("GÜÇLÜ SATICI BASKISI" if satici_yuzde > 55 else "DENGELİ PİYASA")
            b3.metric("⚖️ Piyasa Dengesi", durum_metni, f"Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}")

            st.markdown(f"""
            <div style="width:100%; background:#ef5350; border-radius:8px; height:28px; overflow:hidden; display:flex; margin:10px 0;">
                <div style="width:{alici_yuzde}%; background:#26a69a; height:100%; display:flex; align-items:center; padding-left:12px; font-weight:bold; color:white;">
                    ALICILAR %{alici_yuzde:.1f}
                </div>
                <div style="width:{satici_yuzde}%; height:100%; display:flex; align-items:center; justify-content:flex-end; padding-right:12px; font-weight:bold; color:white;">
                    SATICILAR %{satici_yuzde:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            alis_sirali = alislar.sort_values("fiyat", ascending=False).copy()
            alis_sirali["kumulatif"] = alis_sirali["miktar"].cumsum()
            satis_sirali = satislar.sort_values("fiyat", ascending=True).copy()
            satis_sirali["kumulatif"] = satis_sirali["miktar"].cumsum()

            fig_depth = go.Figure()
            fig_depth.add_trace(go.Scatter(
                x=alis_sirali["fiyat"], y=alis_sirali["kumulatif"],
                fill="tozeroy", mode="lines", line=dict(color="#26a69a", width=2), name="Alış Emirleri (Bid)"
            ))
            fig_depth.add_trace(go.Scatter(
                x=satis_sirali["fiyat"], y=satis_sirali["kumulatif"],
                fill="tozeroy", mode="lines", line=dict(color="#ef5350", width=2), name="Satış Emirleri (Ask)"
            ))
            fig_depth.update_layout(height=420, template="plotly_dark", title="Canlı Emir Defteri Derinlik Grafiği (Depth Chart)", xaxis_title="Fiyat", yaxis_title="Kümülatif Miktar")
            st.plotly_chart(fig_depth, use_container_width=True)

            t1, t2 = st.columns(2)
            with t1:
                st.markdown("#### 🟢 En Büyük 10 Alış Duvarı (Destek)")
                st.dataframe(alislar.sort_values("toplam_usdt", ascending=False).head(10), use_container_width=True, hide_index=True)
            with t2:
                st.markdown("#### 🔴 En Büyük 10 Satış Duvarı (Direnç)")
                st.dataframe(satislar.sort_values("toplam_usdt", ascending=False).head(10), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Emir defteri güncellenirken hata oluştu: {e}")

    canli_emir_defteri_paneli()

# ==============================================================================
# SEKME 3: BIST / FOREX & KÜRESEL MERKEZ BANKALARI HABER AKIŞI
# ==============================================================================
elif st.session_state.aktif_sekme == "haberler":
    st.subheader("🌍 BIST / Forex Piyasaları & Merkez Bankaları Canlı Haber Akışı")

    piyasalar = {
        "BIST 100": "XU100.IS", "USD/TRY": "TRY=X", "EUR/TRY": "EURTRY=X",
        "DXY Endeksi": "DX-Y.NYB", "Ons Altın": "GC=F", "Ons Gümüş": "SI=F",
        "Brent Petrol": "BZ=F", "S&P 500": "^GSPC"
    }
    p_cols = st.columns(4)
    for idx, (ad, kod) in enumerate(piyasalar.items()):
        try:
            h = yf.Ticker(kod).history(period="2d")
            if len(h) >= 2:
                son = float(h["Close"].iloc[-1])
                deg = ((son - float(h["Close"].iloc[-2])) / float(h["Close"].iloc[-2])) * 100
                p_cols[idx % 4].metric(ad, f"{son:,.2f}", f"%{deg:+.2f}")
        except Exception:
            p_cols[idx % 4].metric(ad, "Veri Bekleniyor")

    st.divider()
    haber_kategori = st.radio(
        "Haber Akışı Kategorisi Seçin:",
        ["🪙 Kripto Piyasası", "🇹🇷 TCMB & Türkiye Ekonomi", "🇺🇸 FED & ABD Piyasaları", "🇪🇺 ECB & Euro Bölgesi", "🌐 G20 & Küresel Ekonomi"],
        horizontal=True
    )

    rss_kaynaklari = {
        "🪙 Kripto Piyasası": [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ],
        "🇹🇷 TCMB & Türkiye Ekonomi": [
            "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/RSS",
            "https://www.bloomberght.com/rss"
        ],
        "🇺🇸 FED & ABD Piyasaları": [
            "https://www.federalreserve.gov/feeds/press_all.xml",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        ],
        "🇪🇺 ECB & Euro Bölgesi": [
            "https://www.ecb.europa.eu/rss/press.html"
        ],
        "🌐 G20 & Küresel Ekonomi": [
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://www.cnbc.com/id/100727362/device/rss/rss.html"
        ]
    }

    for url in rss_kaynaklari.get(haber_kategori, []):
        try:
            akis = feedparser.parse(url)
            for haber in akis.entries[:8]:
                baslik = haber.get("title", "Başlık Yok")
                link = haber.get("link", "#")
                tarih = haber.get("published", "")
                st.markdown("**[" + baslik + "](" + link + ")**  \n<small style='color:gray;'>" + tarih + "</small>", unsafe_allow_html=True)
        except Exception:
            continue
