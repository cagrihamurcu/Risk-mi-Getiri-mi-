# app.py
# Borsa Uygulamaları – Bireysel Portföy Oyunu
# (Piyasa Koşulları + Kur + Grafik + Tur Sonu Açıklama)
#
# Güncelleme:
# - "CDS → Faiz → Tahvil Fiyatı (3 adım)" bölümü KALDIRILDI (tam gelmiyordu, gereksiz kalabalık yapıyordu).
# - 1) 2) 3) başlıklarının altına kısa açıklayıcı metin eklendi.
# - Leaderboard yok.
# - matplotlib yok (Streamlit Cloud uyumlu).
# - Sonuçlar tabloları okunur olacak şekilde "Özet/Detay/Açıklamalar" sekmeleriyle veriliyor.

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Risk mi Getiri mi? | Portföy Oyunu", layout="wide")

# -------------------------------------------------
# CSS: (1) Piyasa Kartı altı küçük font
#      (2) Piyasa Koşulları değerini küçük yapan özel kart
# -------------------------------------------------
st.markdown("""
<style>
/* 2) Bu Tur Piyasa Kartı altı */
.piyasa_karti {
  font-size: 0.84rem;
  line-height: 1.25;
}
.piyasa_karti p, .piyasa_karti li, .piyasa_karti .stMarkdown, .piyasa_karti div, .piyasa_karti span {
  font-size: 0.84rem !important;
}
.piyasa_karti [data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
.piyasa_karti [data-testid="stMetricValue"] { font-size: 1.00rem !important; }
.piyasa_karti [data-testid="stDataFrame"] { font-size: 0.78rem !important; }

/* Piyasa Koşulları için özel kart (değer küçük) */
.pk_card {
  border: 1px solid rgba(49, 51, 63, 0.2);
  border-radius: 12px;
  padding: 10px 12px;
  height: 92px;
}
.pk_label {
  font-size: 0.70rem;
  opacity: 0.75;
  margin-bottom: 6px;
}
.pk_value {
  font-size: 0.88rem;   /* değer yazısı küçük */
  font-weight: 700;
  line-height: 1.1;
  word-break: break-word;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# OYUN PARAMETRELERİ
# -------------------------------------------------
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

BASE = {
    "TR": {"mu": 0.020, "sigma": 0.020},
    "US": {"mu": 0.010, "sigma": 0.010},
    "EQ": {"mu": 0.030, "sigma": 0.060},
    "FX": {"mu": 0.015, "sigma": 0.040},
    "CASH": {"mu": 0.000, "sigma": 0.000},
}

PIYASA_KOSULLARI = {
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
    {"tur": 1, "piyasa_kosullari": "Sakin", "haber": "Piyasalarda sakin dönem", "policy": 0.35, "cds": 250, "inf": 0.30},
    {"tur": 2, "piyasa_kosullari": "Enflasyon Baskısı", "haber": "Enflasyon beklentisi yükseldi", "policy": 0.35, "cds": 350, "inf": 0.45},
    {"tur": 3, "piyasa_kosullari": "Risk Şoku", "haber": "Risk algısı bozuldu, CDS yükseldi", "policy": 0.40, "cds": 650, "inf": 0.50},
    {"tur": 4, "piyasa_kosullari": "Stres", "haber": "Belirsizlik artıyor: risk primi yüksek", "policy": 0.45, "cds": 800, "inf": 0.55},
    {"tur": 5, "piyasa_kosullari": "İyileşme", "haber": "Kısmi iyileşme: CDS geriliyor", "policy": 0.40, "cds": 420, "inf": 0.40},
]

# -------------------------------------------------
# FONKSİYONLAR
# -------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def tr_yield(policy: float, cds_bps: int, inf: float) -> float:
    # Temsili TR 2Y faiz
    k = 1.10
    risk_premium = (cds_bps / 10000.0) * k
    inflation_layer = 0.30 * inf
    y = max(policy + inflation_layer + risk_premium, 0.0)
    return y

def bond_price_from_yield(y: float, duration: float = 1.8, face: float = 100.0) -> float:
    y = max(y, 1e-6)
    return face / ((1.0 + y) ** duration)

def dynamic_params(piyasa_kosullari: str, cds_bps: int) -> dict:
    out = {a: {"mu": BASE[a]["mu"], "sigma": BASE[a]["sigma"]} for a in ASSETS}

    r = PIYASA_KOSULLARI[piyasa_kosullari]
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
        r[a] = 0.0 if a == "CASH" else float(rng.normal(dyn[a]["mu"], dyn[a]["sigma"]))
    return r

def validate_total(pcts: dict) -> tuple[bool, str, int]:
    total = int(round(sum(pcts.values())))
    if any(v < 0 for v in pcts.values()):
        return False, "Yüzdeler negatif olamaz.", total
    if total != 100:
        return False, "Toplam **%100** olmalı.", total
    return True, "", total

def portfolio_expected(weights: dict, dyn: dict) -> tuple[float, float]:
    mu = 0.0
    var = 0.0
    for a in ASSETS:
        mu += weights[a] * dyn[a]["mu"]
        var += (weights[a] ** 2) * (dyn[a]["sigma"] ** 2)
    return float(mu), float(np.sqrt(var))

def tur_sonu_aciklama(piyasa_kosullari: str, cds_bps: int, weights: dict, rets: dict, tr_price_effect: float) -> str:
    contributions = {a: weights[a] * rets[a] for a in ASSETS}
    biggest = max(contributions, key=lambda k: abs(contributions[k]))
    biggest_name = ASSET_NAMES[biggest]
    biggest_contrib = contributions[biggest] * 100

    lines = []
    lines.append(f"**Piyasa Koşulları:** {piyasa_kosullari} | **CDS:** {cds_bps} bps")

    if cds_bps >= 650:
        lines.append("- CDS yüksek: risk algısı bozulur → borsa daha oynak, kurda yukarı baskı olasılığı daha yüksek.")
    elif cds_bps >= 400:
        lines.append("- CDS orta-yüksek: risk algısı temkinli → riskli varlıklarda dalgalanma artabilir.")
    else:
        lines.append("- CDS düşük/orta: risk algısı daha sakin → dalgalanma nispeten sınırlı kalabilir.")

    if abs(tr_price_effect) > 0.002:
        direction = "azaldı" if tr_price_effect < 0 else "arttı"
        lines.append(f"- Faiz değişimi nedeniyle TR tahvil fiyatı **{direction}** (fiyat etkisi: **{tr_price_effect*100:.2f}%**).")

    lines.append(f"- Portföy sonucunu en çok etkileyen kalem: **{biggest_name}** (katkı: **{biggest_contrib:+.2f} puan**).")

    if weights["EQ"] + weights["FX"] >= 0.60:
        lines.append("- Portföy riskli ağırlıklı (Borsa + Kur yüksek): sonuçlar daha değişken olabilir.")
    elif weights["TR"] + weights["US"] + weights["CASH"] >= 0.70:
        lines.append("- Portföy daha korumacı (Tahvil + Nakit yüksek): dalgalanma genelde daha sınırlı olur.")
    else:
        lines.append("- Portföy dengeli: şoklarda darbeyi azaltıp iyileşmede fırsat yakalama potansiyeli artar.")

    return "\n".join(lines)

def pk_card_html(label: str, value: str) -> str:
    return f"""
<div class="pk_card">
  <div class="pk_label">{label}</div>
  <div class="pk_value">{value}</div>
</div>
"""

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "player" not in st.session_state:
    st.session_state.player = "Öğrenci"
if "capital" not in st.session_state:
    st.session_state.capital = float(STARTING_CAPITAL)
if "tur_idx" not in st.session_state:
    st.session_state.tur_idx = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "seed" not in st.session_state:
    st.session_state.seed = 42
if "prev_tr_yield" not in st.session_state:
    first = ROUNDS[0]
    st.session_state.prev_tr_yield = tr_yield(first["policy"], first["cds"], first["inf"])

for k, v in [("pct_tr", 35), ("pct_us", 20), ("pct_eq", 30), ("pct_fx", 10), ("pct_cash", 5)]:
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------------------------
# ÜST
# -------------------------------------------------
st.title("🎮 Risk mi Getiri mi? – Bireysel Portföy Oyunu")
st.caption("Piyasa Koşulları değiştikçe CDS–faiz–tahvil fiyatı–borsa–kur dinamikleri farklılaşır.")

a, b, c = st.columns([1.1, 1.0, 1.0])
with a:
    st.session_state.player = st.text_input("Oyuncu adı", value=st.session_state.player, max_chars=30)
with b:
    st.metric("Portföy Değeri", f"{st.session_state.capital:,.0f} TL")
with c:
    if st.button("🔄 Oyunu Sıfırla"):
        st.session_state.capital = float(STARTING_CAPITAL)
        st.session_state.tur_idx = 0
        st.session_state.history = []
        first = ROUNDS[0]
        st.session_state.prev_tr_yield = tr_yield(first["policy"], first["cds"], first["inf"])
        st.session_state.pct_tr, st.session_state.pct_us, st.session_state.pct_eq, st.session_state.pct_fx, st.session_state.pct_cash = 35, 20, 30, 10, 5
        st.rerun()

st.divider()
left, right = st.columns([1.25, 0.75])

# -------------------------------------------------
# SOL
# -------------------------------------------------
with left:
    st.subheader("1) Varlık Kartları (Baz)")
    st.caption("Bu tablo, her varlık için **başlangıç (normal dönem)** beklenen getiri ve oynaklık varsayımlarını gösterir.")

    df_cards = pd.DataFrame(
        [{
            "Varlık": ASSET_NAMES[a],
            "Baz beklenen getiri (tur)": f"{BASE[a]['mu']*100:.1f}%",
            "Baz oynaklık (tur)": f"{BASE[a]['sigma']*100:.1f}%"
        } for a in ASSETS]
    )
    st.dataframe(df_cards, use_container_width=True)

    st.subheader("2) Bu Tur Piyasa Kartı")
    st.caption("Bu turda **Piyasa Koşulları + CDS + politika faizi + enflasyon beklentisi** birlikte çalışır ve varlıkların risk/getiri profilini değiştirir.")

    # Başlık normal, altı küçük font
    st.markdown('<div class="piyasa_karti">', unsafe_allow_html=True)

    if st.session_state.tur_idx >= N_ROUNDS:
        st.success("Oyun tamamlandı ✅ (Aşağıda sonuçlar var.)")
    else:
        r = ROUNDS[st.session_state.tur_idx]
        tr_y = tr_yield(r["policy"], r["cds"], r["inf"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tur", f"{r['tur']}/{N_ROUNDS}")
        c2.markdown(pk_card_html("Piyasa Koşulları", r["piyasa_kosullari"]), unsafe_allow_html=True)
        c3.metric("Politika", f"%{r['policy']*100:.1f}")
        c4.metric("CDS", f"{r['cds']} bps")
        c5.metric("Enflasyon", f"%{r['inf']*100:.1f}")

        st.write(f"**Haber:** {r['haber']}")
        st.metric("TR 2Y tahvil faizi (temsili)", f"%{tr_y*100:.1f}")

        dyn = dynamic_params(r["piyasa_kosullari"], r["cds"])
        df_dyn = pd.DataFrame([{
            "Varlık": ASSET_NAMES[a],
            "Beklenen getiri (bu tur)": f"{dyn[a]['mu']*100:.2f}%",
            "Oynaklık (bu tur)": f"{dyn[a]['sigma']*100:.2f}%"
        } for a in ASSETS])
        st.dataframe(df_dyn, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# SAĞ
# -------------------------------------------------
with right:
    st.subheader("3) Karar Ver ve Oyna")
    st.caption("Portföy yüzdelerini seç ve turu oynat. Oyun sonunda her tur için **getiri, portföy değeri ve kısa açıklama** oluşur.")

    if st.session_state.tur_idx >= N_ROUNDS:
        st.success("Oyun bitti ✅ Aşağıda sonuçlar.")
    else:
        r = ROUNDS[st.session_state.tur_idx]
        tr_y = tr_yield(r["policy"], r["cds"], r["inf"])
        dyn = dynamic_params(r["piyasa_kosullari"], r["cds"])

        st.write("Yüzdeleri gir (toplam %100):")

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

        if st.button("▶️ Turu Oyna", disabled=(not ok)):
            rng = np.random.default_rng(st.session_state.seed + r["tur"] * 101)

            # Tahvil fiyat etkisi: faiz değişimi (basit endeks)
            prev_y = st.session_state.prev_tr_yield
            prev_p = bond_price_from_yield(prev_y)
            curr_p = bond_price_from_yield(tr_y)
            tr_price_effect = (curr_p - prev_p) / prev_p

            rets = simulate_returns(rng, dyn)
            rets["TR"] = float(rets["TR"] + tr_price_effect)

            port_r = sum(weights[a] * rets[a] for a in ASSETS)
            new_val = float(st.session_state.capital * (1.0 + port_r))
            st.session_state.capital = new_val

            explanation = tur_sonu_aciklama(
                piyasa_kosullari=r["piyasa_kosullari"],
                cds_bps=r["cds"],
                weights=weights,
                rets=rets,
                tr_price_effect=tr_price_effect
            )

            st.session_state.history.append({
                "Tur": r["tur"],
                "Piyasa Koşulları": r["piyasa_kosullari"],
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
                "Açıklama": explanation
            })

            st.session_state.prev_tr_yield = tr_y
            st.session_state.tur_idx += 1
            st.rerun()

st.divider()

# -------------------------------------------------
# SONUÇLAR (Okunur)
# -------------------------------------------------
st.subheader("📊 Sonuçlar")

if len(st.session_state.history) == 0:
    st.write("Henüz tur oynanmadı.")
else:
    df = pd.DataFrame(st.session_state.history)

    tab1, tab2, tab3 = st.tabs(["Özet Tablo (okunur)", "Detay Tablo (kaydır)", "Açıklamalar"])

    with tab1:
        df_sum = df[["Tur", "Piyasa Koşulları", "CDS", "TR_Faiz", "Portföy_Getiri", "Portföy_Değeri"]].copy()
        df_sum["TR_Faiz"] = (df_sum["TR_Faiz"] * 100).round(1)
        df_sum["Portföy_Getiri"] = (df_sum["Portföy_Getiri"] * 100).round(2)
        df_sum["Portföy_Değeri"] = df_sum["Portföy_Değeri"].round(0).astype(int)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

    with tab2:
        df_det = df.drop(columns=["Açıklama"]).copy()
        df_det["TR_Faiz"] = (df_det["TR_Faiz"] * 100).round(1)
        df_det["Tahvil_Fiyat_Etkisi"] = (df_det["Tahvil_Fiyat_Etkisi"] * 100).round(2)
        for c in ["TR_Getiri", "US_Getiri", "Borsa_Getiri", "Kur_Getiri", "Portföy_Getiri"]:
            df_det[c] = (df_det[c] * 100).round(2)
        for c in ["A_TR", "A_US", "A_EQ", "A_FX", "A_CASH"]:
            df_det[c] = (df_det[c] * 100).round(0).astype(int)
        st.dataframe(df_det, use_container_width=True, hide_index=True)

    with tab3:
        for _, row in df.iterrows():
            with st.expander(f"Tur {int(row['Tur'])} – {row['Piyasa Koşulları']}", expanded=False):
                st.markdown(row["Açıklama"])

    st.subheader("📈 Portföy Değeri Grafiği")
    st.line_chart(df[["Tur", "Portföy_Değeri"]].set_index("Tur"))

    st.download_button(
        "⬇️ Sonuçları CSV indir",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="borsa_portfoy_oyunu_sonuclar.csv",
        mime="text/csv",
    )
