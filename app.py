# app.py
# Borsa Uygulamaları – Bireysel Portföy Oyunu (Rejim Bazlı + Kur + Leaderboard + Grafik + Mekanizma Animasyonu)
#
# Eklenenler:
# 1) Leaderboard (aynı cihaz/oturum içinde isim bazlı skor tablosu)
# 2) Portföy grafiği (tur bazında değer)
# 3) CDS → Faiz → Tahvil Fiyatı mekanizması için küçük “animasyon” (adım adım gösterim)
#
# Notlar:
# - Kuyruk riski ve VaR YOK.
# - Leaderboard kalıcı değildir (Streamlit oturumu kapanınca sıfırlanır). İstersen CSV kaydı ekleyebilirim.
#
# Çalıştır:
#   pip install streamlit numpy pandas matplotlib
#   streamlit run app.py

import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Risk mi Getiri mi? | Portföy Oyunu", layout="wide")

# -----------------------------
# CSS: Piyasa Kartı font küçültme
# -----------------------------
st.markdown(
    """
<style>
/* Sadece "piyasa_karti" alanını küçült */
.piyasa_karti { font-size: 0.85rem; line-height: 1.25; }
.piyasa_karti p, .piyasa_karti li, .piyasa_karti .stMarkdown { font-size: 0.85rem !important; }
.piyasa_karti [data-testid="stMetricLabel"] { font-size: 0.74rem !important; }
.piyasa_karti [data-testid="stMetricValue"] { font-size: 1.02rem !important; }
.piyasa_karti [data-testid="stDataFrame"] { font-size: 0.78rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Oyun Parametreleri
# -----------------------------
STARTING_CAPITAL = 100_000
N_ROUNDS = 5

ASSETS = ["TR", "US", "EQ", "FX", "CASH"]
ASSET_NAMES = {
    "TR": "Türkiye 2Y Tahvil",
    "US": "ABD Tahvil (Düşük risk)",
    "EQ": "Borsa Endeksi",
    "FX": "USD/TRY (Kur)",
    "CASH": "Nakit",
}

# Baz (tur başına) temsili
BASE = {
    "TR": {"mu": 0.020, "sigma": 0.020},
    "US": {"mu": 0.010, "sigma": 0.010},
    "EQ": {"mu": 0.030, "sigma": 0.060},
    "FX": {"mu": 0.015, "sigma": 0.040},
    "CASH": {"mu": 0.000, "sigma": 0.000},
}

REGIMES = {
    "Sakin": {
        "TR": {"mu_add": 0.000, "sigma_mult": 1.00},
        "US": {"mu_add": 0.000, "sigma_mult": 1.00},
        "EQ": {"mu_add": 0.000, "sigma_mult": 1.00},
        "FX": {"mu_add": 0.000, "sigma_mult": 0.95},
    },
    "Enflasyon Baskısı": {
        "TR": {"mu_add": -0.002, "sigma_mult": 1.20},
        "US": {"mu_add": 0.000, "sigma_mult": 1.05},
        "EQ": {"mu_add": -0.004, "sigma_mult": 1.15},
        "FX": {"mu_add": 0.004, "sigma_mult": 1.15},
    },
    "Risk Şoku": {
        "TR": {"mu_add": -0.010, "sigma_mult": 1.60},
        "US": {"mu_add": 0.001, "sigma_mult": 1.10},
        "EQ": {"mu_add": -0.015, "sigma_mult": 1.45},
        "FX": {"mu_add": 0.012, "sigma_mult": 1.60},
    },
    "Stres": {
        "TR": {"mu_add": -0.006, "sigma_mult": 1.35},
        "US": {"mu_add": 0.001, "sigma_mult": 1.10},
        "EQ": {"mu_add": -0.010, "sigma_mult": 1.30},
        "FX": {"mu_add": 0.008, "sigma_mult": 1.35},
    },
    "İyileşme": {
        "TR": {"mu_add": 0.003, "sigma_mult": 0.95},
        "US": {"mu_add": 0.000, "sigma_mult": 1.00},
        "EQ": {"mu_add": 0.006, "sigma_mult": 0.90},
        "FX": {"mu_add": -0.004, "sigma_mult": 0.95},
    },
}

ROUNDS = [
    {"tur": 1, "rejim": "Sakin", "haber": "Piyasalarda sakin dönem", "policy": 0.35, "cds": 250, "inf": 0.30},
    {"tur": 2, "rejim": "Enflasyon Baskısı", "haber": "Enflasyon beklentisi yükseldi", "policy": 0.35, "cds": 350, "inf": 0.45},
    {"tur": 3, "rejim": "Risk Şoku", "haber": "Risk algısı bozuldu, CDS yükseldi", "policy": 0.40, "cds": 650, "inf": 0.50},
    {"tur": 4, "rejim": "Stres", "haber": "Belirsizlik artıyor: risk primi yüksek", "policy": 0.45, "cds": 800, "inf": 0.55},
    {"tur": 5, "rejim": "İyileşme", "haber": "Kısmi iyileşme: CDS geriliyor", "policy": 0.40, "cds": 420, "inf": 0.40},
]

# -----------------------------
# Fonksiyonlar
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def tr_yield_components(policy: float, cds_bps: int, inf: float) -> dict:
    """
    TR 2Y faiz (temsili):
      y = politika + (enflasyon beklentisi katmanı) + risk primi(CDS)
    """
    k = 1.10
    risk_premium = (cds_bps / 10000.0) * k
    inflation_layer = 0.30 * inf
    y = max(policy + inflation_layer + risk_premium, 0.0)
    return {"yield": y, "policy": policy, "infl_layer": inflation_layer, "risk_premium": risk_premium}

def bond_price_from_yield(y: float, duration: float = 1.8, face: float = 100.0) -> float:
    y = max(y, 1e-6)
    return face / ((1.0 + y) ** duration)

def dynamic_params(regime: str, cds_bps: int) -> dict:
    """
    Rejim + CDS'e göre mu/sigma.
    CDS ↑:
      - EQ mu ↓, sigma ↑
      - TR sigma ↑, mu hafif ↓
      - FX mu ↑, sigma ↑
    """
    out = {a: {"mu": BASE[a]["mu"], "sigma": BASE[a]["sigma"]} for a in ASSETS}

    r = REGIMES[regime]
    for a in ["TR", "US", "EQ", "FX"]:
        out[a]["mu"] += r[a]["mu_add"]
        out[a]["sigma"] *= r[a]["sigma_mult"]

    cds_scale = cds_bps / 10000.0

    out["EQ"]["mu"] -= cds_scale * 0.08
    out["EQ"]["sigma"] *= (1.0 + cds_scale * 1.2)

    out["TR"]["mu"] -= cds_scale * 0.02
    out["TR"]["sigma"] *= (1.0 + cds_scale * 1.0)

    out["FX"]["mu"] += cds_scale * 0.10
    out["FX"]["sigma"] *= (1.0 + cds_scale * 1.6)

    out["EQ"]["mu"] = clamp(out["EQ"]["mu"], -0.06, 0.06)
    out["TR"]["mu"] = clamp(out["TR"]["mu"], -0.03, 0.05)
    out["FX"]["mu"] = clamp(out["FX"]["mu"], -0.02, 0.08)

    return out

def simulate_returns(rng: np.random.Generator, dyn: dict) -> dict:
    r = {}
    for a in ASSETS:
        if a == "CASH":
            r[a] = 0.0
        else:
            r[a] = float(rng.normal(dyn[a]["mu"], dyn[a]["sigma"]))
    return r

def validate_total(pcts: dict) -> tuple[bool, str, int]:
    total = int(round(sum(pcts.values())))
    if any(v < 0 for v in pcts.values()):
        return False, "Yüzdeler negatif olamaz.", total
    if total != 100:
        return False, "Toplam **%100** olmalı.", total
    return True, "", total

def portfolio_expected(weights: dict, dyn: dict) -> tuple[float, float]:
    # korelasyon yok varsayımı (ders için)
    mu = 0.0
    var = 0.0
    for a in ASSETS:
        mu += weights[a] * dyn[a]["mu"]
        var += (weights[a] ** 2) * (dyn[a]["sigma"] ** 2)
    return float(mu), float(np.sqrt(var))

def run_mechanism_animation(container, cds_bps: int, policy: float, inf: float, prev_y: float):
    """
    Adım adım küçük animasyon: CDS → faiz → fiyat
    """
    # Adım 1: CDS
    container.info(f"Adım 1/3: CDS = **{cds_bps} bps** → risk primi artabilir.")
    time.sleep(0.35)

    comps = tr_yield_components(policy, cds_bps, inf)
    y = comps["yield"]

    # Adım 2: Faiz
    container.warning(
        f"Adım 2/3: TR 2Y faiz ≈ politika + enflasyon katmanı + risk primi\n\n"
        f"- Politika: **%{comps['policy']*100:.1f}**\n"
        f"- Enflasyon katmanı: **%{comps['infl_layer']*100:.1f}**\n"
        f"- Risk primi: **%{comps['risk_premium']*100:.1f}**\n\n"
        f"➡️ Toplam faiz: **%{y*100:.1f}**"
    )
    time.sleep(0.35)

    # Adım 3: Fiyat
    prev_p = bond_price_from_yield(prev_y)
    curr_p = bond_price_from_yield(y)
    price_effect = (curr_p - prev_p) / prev_p

    container.error(
        f"Adım 3/3: Faiz ↑ ise tahvil fiyatı ↓ eğiliminde\n\n"
        f"- Önceki faiz: **%{prev_y*100:.1f}** → fiyat endeksi: **{prev_p:.2f}**\n"
        f"- Yeni faiz: **%{y*100:.1f}** → fiyat endeksi: **{curr_p:.2f}**\n"
        f"➡️ Tahvil fiyat etkisi: **{price_effect*100:.2f}%**"
    )

# -----------------------------
# Session State
# -----------------------------
if "player" not in st.session_state:
    st.session_state.player = "Öğrenci"

if "capital" not in st.session_state:
    st.session_state.capital = float(STARTING_CAPITAL)

if "tur_idx" not in st.session_state:
    st.session_state.tur_idx = 0

if "history" not in st.session_state:
    st.session_state.history = []  # tur bazlı

if "seed" not in st.session_state:
    st.session_state.seed = 42  # sabit (kontrol paneli yok)

if "prev_tr_yield" not in st.session_state:
    first = ROUNDS[0]
    st.session_state.prev_tr_yield = tr_yield_components(first["policy"], first["cds"], first["inf"])["yield"]

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []  # dict: name, final_value, date/time optional

# Varsayılan yüzdeler
for k, v in [("pct_tr", 35), ("pct_us", 20), ("pct_eq", 30), ("pct_fx", 10), ("pct_cash", 5)]:
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Üst Başlık
# -----------------------------
st.title("🎮 Risk mi Getiri mi? – Bireysel Portföy Oyunu (Kur + Grafik + Leaderboard)")
st.caption("Rejim bazlı senaryo: CDS–faiz–tahvil fiyatı mekanizması + borsa + kur.")

top_a, top_b, top_c = st.columns([1.1, 1.0, 1.0])
with top_a:
    st.session_state.player = st.text_input("Oyuncu adı", value=st.session_state.player, max_chars=30)
with top_b:
    st.metric("Portföy Değeri", f"{st.session_state.capital:,.0f} TL")
with top_c:
    if st.button("🔄 Oyunu Sıfırla"):
        st.session_state.capital = float(STARTING_CAPITAL)
        st.session_state.tur_idx = 0
        st.session_state.history = []
        first = ROUNDS[0]
        st.session_state.prev_tr_yield = tr_yield_components(first["policy"], first["cds"], first["inf"])["yield"]
        st.session_state.pct_tr, st.session_state.pct_us, st.session_state.pct_eq, st.session_state.pct_fx, st.session_state.pct_cash = 35, 20, 30, 10, 5
        st.rerun()

st.divider()

left, right = st.columns([1.25, 0.75])

# -----------------------------
# SOL: Varlık kartları + Piyasa kartı (font küçük)
# -----------------------------
with left:
    st.subheader("1) Varlık Kartları (Baz)")

    df_cards = pd.DataFrame(
        [{
            "Varlık": ASSET_NAMES[a],
            "Baz beklenen getiri (tur)": f"{BASE[a]['mu']*100:.1f}%",
            "Baz oynaklık (tur)": f"{BASE[a]['sigma']*100:.1f}%"
        } for a in ASSETS]
    )
    st.dataframe(df_cards, use_container_width=True)

    st.subheader("2) Bu Tur Piyasa Kartı")
    st.markdown('<div class="piyasa_karti">', unsafe_allow_html=True)

    if st.session_state.tur_idx >= N_ROUNDS:
        st.success("Oyun tamamlandı ✅ (Aşağıda sonuçlar ve leaderboard var.)")
    else:
        r = ROUNDS[st.session_state.tur_idx]
        comps = tr_yield_components(r["policy"], r["cds"], r["inf"])
        tr_y = comps["yield"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tur", f"{r['tur']}/{N_ROUNDS}")
        c2.metric("Rejim", r["rejim"])
        c3.metric("Politika", f"%{r['policy']*100:.1f}")
        c4.metric("CDS", f"{r['cds']} bps")
        c5.metric("Enflasyon", f"%{r['inf']*100:.1f}")

        st.write(f"**Haber:** {r['haber']}")
        st.metric("TR 2Y tahvil faizi (temsili)", f"%{tr_y*100:.1f}")

        dyn = dynamic_params(r["rejim"], r["cds"])
        df_dyn = pd.DataFrame([{
            "Varlık": ASSET_NAMES[a],
            "Beklenen getiri (bu tur)": f"{dyn[a]['mu']*100:.2f}%",
            "Oynaklık (bu tur)": f"{dyn[a]['sigma']*100:.2f}%"
        } for a in ASSETS])
        st.dataframe(df_dyn, use_container_width=True)

        with st.expander("CDS → Faiz → Tahvil Fiyatı (Animasyon)", expanded=False):
            ph = st.empty()
            if st.button("▶️ Mekanizmayı Göster (3 adım)"):
                run_mechanism_animation(
                    container=ph,
                    cds_bps=r["cds"],
                    policy=r["policy"],
                    inf=r["inf"],
                    prev_y=st.session_state.prev_tr_yield
                )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# SAĞ: Karar + Oyna
# -----------------------------
with right:
    st.subheader("3) Karar Ver ve Oyna")

    if st.session_state.tur_idx >= N_ROUNDS:
        st.success("Oyun bitti ✅ Aşağıda sonuçlar/leaderboard.")
    else:
        r = ROUNDS[st.session_state.tur_idx]
        comps = tr_yield_components(r["policy"], r["cds"], r["inf"])
        tr_y = comps["yield"]
        dyn = dynamic_params(r["rejim"], r["cds"])

        st.write("Yüzdeleri kutucuklardan gir (toplam %100):")

        i1, i2 = st.columns(2)
        with i1:
            pct_tr = st.number_input("TR Tahvil (%)", 0, 100, int(st.session_state.pct_tr), 1, key="pct_tr")
            pct_eq = st.number_input("Borsa (%)", 0, 100, int(st.session_state.pct_eq), 1, key="pct_eq")
            pct_cash = st.number_input("Nakit (%)", 0, 100, int(st.session_state.pct_cash), 1, key="pct_cash")
        with i2:
            pct_us = st.number_input("US Tahvil (%)", 0, 100, int(st.session_state.pct_us), 1, key="pct_us")
            pct_fx = st.number_input("USD/TRY (%)", 0, 100, int(st.session_state.pct_fx), 1, key="pct_fx")

        pcts = {"TR": pct_tr, "US": pct_us, "EQ": pct_eq, "FX": pct_fx, "CASH": pct_cash}
        ok, msg, total = validate_total(pcts)
        st.write(f"Toplam: **%{total}**")
        if not ok:
            st.error(msg)

        weights = {a: pcts[a] / 100.0 for a in ASSETS}
        exp_mu, exp_sigma = portfolio_expected(weights, dyn)

        m1, m2 = st.columns(2)
        m1.metric("Beklenen getiri (bu tur)", f"{exp_mu*100:.2f}%")
        m2.metric("Tahmini risk (bu tur)", f"{exp_sigma*100:.2f}%")

        # Tur oyna
        if st.button("▶️ Turu Oyna", disabled=(not ok)):
            # Tahvil fiyat etkisi (faiz değişimi)
            prev_y = st.session_state.prev_tr_yield
            prev_p = bond_price_from_yield(prev_y)
            curr_p = bond_price_from_yield(tr_y)
            tr_price_effect = (curr_p - prev_p) / prev_p

            rng = np.random.default_rng(st.session_state.seed + r["tur"] * 101)

            # Getirileri üret
            rets = simulate_returns(rng, dyn)

            # TR tahvil getirisini faiz->fiyat etkisi ile düzelt
            rets["TR"] = float(rets["TR"] + tr_price_effect)

            # Portföy getirisi
            port_r = 0.0
            for a in ASSETS:
                port_r += weights[a] * rets[a]

            old_val = st.session_state.capital
            new_val = old_val * (1.0 + port_r)
            st.session_state.capital = float(new_val)

            st.session_state.history.append({
                "Tur": r["tur"],
                "Rejim": r["rejim"],
                "CDS": r["cds"],
                "TR_Faiz": tr_y,
                "Tahvil_Fiyat_Etkisi": tr_price_effect,
                "TR_Getiri": rets["TR"],
                "US_Getiri": rets["US"],
                "Borsa_Getiri": rets["EQ"],
                "Kur_Getiri": rets["FX"],
                "Portföy_Getiri": port_r,
                "Portföy_Değeri": new_val,
                "A_TR": weights["TR"],
                "A_US": weights["US"],
                "A_EQ": weights["EQ"],
                "A_FX": weights["FX"],
                "A_CASH": weights["CASH"],
            })

            # tur ilerlet
            st.session_state.prev_tr_yield = tr_y
            st.session_state.tur_idx += 1

            st.rerun()

st.divider()

# -----------------------------
# SONUÇLAR: Tablo + Grafik + Leaderboard
# -----------------------------
st.subheader("📊 Sonuçlar")

if len(st.session_state.history) == 0:
    st.write("Henüz tur oynanmadı.")
else:
    df_hist = pd.DataFrame(st.session_state.history)

    # Görüntüleme formatı
    df_show = df_hist.copy()
    df_show["TR_Faiz"] = df_show["TR_Faiz"] * 100
    df_show["Tahvil_Fiyat_Etkisi"] = df_show["Tahvil_Fiyat_Etkisi"] * 100
    for c in ["TR_Getiri", "US_Getiri", "Borsa_Getiri", "Kur_Getiri", "Portföy_Getiri"]:
        df_show[c] = df_show[c] * 100
    for c in ["A_TR", "A_US", "A_EQ", "A_FX", "A_CASH"]:
        df_show[c] = df_show[c] * 100

    st.dataframe(
        df_show.style.format({
            "TR_Faiz": "{:.1f}%",
            "Tahvil_Fiyat_Etkisi": "{:.2f}%",
            "TR_Getiri": "{:.2f}%",
            "US_Getiri": "{:.2f}%",
            "Borsa_Getiri": "{:.2f}%",
            "Kur_Getiri": "{:.2f}%",
            "Portföy_Getiri": "{:.2f}%",
            "Portföy_Değeri": "{:,.0f}",
            "A_TR": "{:.0f}%",
            "A_US": "{:.0f}%",
            "A_EQ": "{:.0f}%",
            "A_FX": "{:.0f}%",
            "A_CASH": "{:.0f}%",
        }),
        use_container_width=True
    )

    # Portföy grafiği
    st.subheader("📈 Portföy Değeri Grafiği")
    fig = plt.figure()
    x = df_hist["Tur"].tolist()
    y = df_hist["Portföy_Değeri"].tolist()
    plt.plot(x, y, marker="o")
    plt.xlabel("Tur")
    plt.ylabel("Portföy Değeri (TL)")
    plt.title("Tur Bazında Portföy Değeri")
    st.pyplot(fig)

    # Oyun bitince leaderboard’a ekleme
    if st.session_state.tur_idx >= N_ROUNDS:
        st.subheader("🏁 Final Skor")

        final_value = float(st.session_state.capital)
        st.metric("Final Portföy Değeri", f"{final_value:,.0f} TL")

        # aynı isimle tekrar eklenmesin diye: sadece bu turda ekle butonu
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("➕ Leaderboard'a Kaydet"):
                st.session_state.leaderboard.append({
                    "Oyuncu": st.session_state.player.strip() or "Öğrenci",
                    "Final": final_value
                })
                st.success("Kaydedildi.")
        with c2:
            if st.button("🧹 Leaderboard'ı Temizle"):
                st.session_state.leaderboard = []
                st.success("Leaderboard temizlendi.")

        # Leaderboard tablosu
        st.subheader("🏆 Leaderboard")
        if len(st.session_state.leaderboard) == 0:
            st.write("Leaderboard boş.")
        else:
            lb = pd.DataFrame(st.session_state.leaderboard)
            lb = lb.sort_values("Final", ascending=False).reset_index(drop=True)
            lb.index = lb.index + 1
            st.dataframe(lb.style.format({"Final": "{:,.0f}"}), use_container_width=True)

    # CSV indir
    st.download_button(
        "⬇️ Sonuçları CSV indir",
        data=df_hist.to_csv(index=False).encode("utf-8"),
        file_name="borsa_portfoy_oyunu_sonuclar.csv",
        mime="text/csv",
    )
