# app.py
# Borsa Uygulamaları – Bireysel Portföy Oyunu (Rejim Bazlı + Kur Varlığı Eklendi)
# CDS – Politika faizi – Enflasyon beklentisi → TR tahvil faizi
# Faiz ↑ → Tahvil fiyatı ↓ (fiyat etkisi)
# Ek: USD/TRY (Kur) varlığı eklendi → CDS şoklarında “Türkiye hissi” daha gerçekçi.
#
# Not: Kuyruk riski ve VaR yok.
#
# Çalıştır:
#   pip install streamlit numpy pandas
#   streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Risk mi Getiri mi? | Bireysel Portföy Oyunu", layout="wide")

# -----------------------------
# Oyun Ayarları
# -----------------------------
N_ROUNDS = 5
STARTING_CAPITAL = 100_000

ASSETS = ["TR_2Y_BOND", "US_BOND", "EQUITY", "FX", "CASH"]
ASSET_NAMES = {
    "TR_2Y_BOND": "Türkiye 2Y Tahvil",
    "US_BOND": "ABD Tahvil (Düşük risk)",
    "EQUITY": "Borsa Endeksi (Riskli)",
    "FX": "USD/TRY (Kur)",
    "CASH": "Nakit",
}

# Baz parametreler (tur başına, temsili)
# FX: CDS ↑ dönemlerinde yukarı eğilim (pozitif getiri) + yüksek oynaklık kurgusu
BASE = {
    "TR_2Y_BOND": {"mu": 0.020, "sigma": 0.015},
    "US_BOND":    {"mu": 0.008, "sigma": 0.005},
    "EQUITY":     {"mu": 0.030, "sigma": 0.060},
    "FX":         {"mu": 0.015, "sigma": 0.040},
    "CASH":       {"mu": 0.000, "sigma": 0.000},
}

# Rejim etkileri (mu ekleri + sigma çarpanları)
REGIMES = {
    "Sakin": {
        "TR_2Y_BOND": {"mu_add": 0.000,  "sigma_mult": 1.00},
        "US_BOND":    {"mu_add": 0.000,  "sigma_mult": 1.00},
        "EQUITY":     {"mu_add": 0.000,  "sigma_mult": 1.00},
        "FX":         {"mu_add": 0.000,  "sigma_mult": 0.95},
    },
    "Enflasyon Baskısı": {
        "TR_2Y_BOND": {"mu_add": -0.002, "sigma_mult": 1.20},
        "US_BOND":    {"mu_add": 0.000,  "sigma_mult": 1.05},
        "EQUITY":     {"mu_add": -0.004, "sigma_mult": 1.15},
        "FX":         {"mu_add": 0.004,  "sigma_mult": 1.15},
    },
    "Risk Şoku": {
        "TR_2Y_BOND": {"mu_add": -0.010, "sigma_mult": 1.60},
        "US_BOND":    {"mu_add": 0.001,  "sigma_mult": 1.10},
        "EQUITY":     {"mu_add": -0.015, "sigma_mult": 1.45},
        "FX":         {"mu_add": 0.012,  "sigma_mult": 1.60},
    },
    "Stres": {
        "TR_2Y_BOND": {"mu_add": -0.006, "sigma_mult": 1.35},
        "US_BOND":    {"mu_add": 0.001,  "sigma_mult": 1.10},
        "EQUITY":     {"mu_add": -0.010, "sigma_mult": 1.30},
        "FX":         {"mu_add": 0.008,  "sigma_mult": 1.35},
    },
    "İyileşme": {
        "TR_2Y_BOND": {"mu_add": 0.003,  "sigma_mult": 0.95},
        "US_BOND":    {"mu_add": 0.000,  "sigma_mult": 1.00},
        "EQUITY":     {"mu_add": 0.006,  "sigma_mult": 0.90},
        "FX":         {"mu_add": -0.004, "sigma_mult": 0.95},
    },
}

# Senaryo turları
ROUNDS = [
    {"tur": 1, "rejim": "Sakin",             "haber": "Sakin dönem: risk algısı düşük",                        "policy": 0.35, "cds": 250, "inf_exp": 0.30},
    {"tur": 2, "rejim": "Enflasyon Baskısı", "haber": "Enflasyon beklentisi yükseliyor",                       "policy": 0.35, "cds": 320, "inf_exp": 0.45},
    {"tur": 3, "rejim": "Risk Şoku",         "haber": "Risk algısı bozuluyor: CDS yükseldi",                   "policy": 0.40, "cds": 650, "inf_exp": 0.50},
    {"tur": 4, "rejim": "Stres",             "haber": "Belirsizlik sürüyor: risk primi yüksek",                "policy": 0.45, "cds": 800, "inf_exp": 0.55},
    {"tur": 5, "rejim": "İyileşme",          "haber": "Kısmi iyileşme: CDS düşüyor, beklentiler toparlanıyor", "policy": 0.40, "cds": 420, "inf_exp": 0.40},
]

# -----------------------------
# Yardımcı Fonksiyonlar
# -----------------------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def tr_yield_from_policy_cds_infl(policy: float, cds_bps: int, inf_exp: float) -> dict:
    k = 1.10
    risk_premium = (cds_bps / 10000.0) * k
    inflation_layer = 0.30 * inf_exp
    y = max(policy + inflation_layer + risk_premium, 0.0)
    return {"yield": y, "policy": policy, "infl_layer": inflation_layer, "risk_premium": risk_premium}

def bond_price_from_yield(y: float, duration: float = 1.8, face: float = 100.0) -> float:
    y = max(y, 1e-6)
    return face / ((1.0 + y) ** duration)

def validate_percentages(pcts: dict) -> tuple[bool, str, int]:
    total = int(round(sum(pcts.values())))
    if total != 100:
        return False, "Toplam **%100** olmalı. (Kutulardan birini azaltıp/artırın.)", total
    if any(v < 0 for v in pcts.values()):
        return False, "Yüzdeler negatif olamaz.", total
    return True, "", total

def portfolio_stats(weights: dict, means: dict, sigmas: dict) -> tuple[float, float]:
    mu_p = sum(weights[a] * means[a] for a in ASSETS)
    var_p = sum((weights[a] ** 2) * (sigmas[a] ** 2) for a in ASSETS)  # korelasyon yok varsayımı (ders için)
    return float(mu_p), float(np.sqrt(var_p))

def get_dynamic_params(regime: str, cds_bps: int) -> dict:
    """
    Rejim + CDS'e göre varlık mu/sigma üret.
    - CDS ↑: Borsa mu ↓, sigma ↑
    - CDS ↑: TR tahvil sigma ↑, mu hafif ↓
    - CDS ↑: Kur (USD/TRY) mu ↑, sigma ↑  (risk arttıkça kur baskısı)
    """
    out = {a: {"mu": BASE[a]["mu"], "sigma": BASE[a]["sigma"]} for a in ASSETS}

    r = REGIMES[regime]
    for a in ["TR_2Y_BOND", "US_BOND", "EQUITY", "FX"]:
        out[a]["mu"] += r[a]["mu_add"]
        out[a]["sigma"] *= r[a]["sigma_mult"]

    cds_scale = cds_bps / 10000.0  # 650 bps -> 0.065

    # Equity CDS etkisi
    out["EQUITY"]["mu"] -= cds_scale * 0.08
    out["EQUITY"]["sigma"] *= (1.0 + cds_scale * 1.2)

    # TR bond CDS etkisi
    out["TR_2Y_BOND"]["mu"] -= cds_scale * 0.02
    out["TR_2Y_BOND"]["sigma"] *= (1.0 + cds_scale * 1.0)

    # FX CDS etkisi (CDS ↑ → kur yukarı baskı + vol ↑)
    out["FX"]["mu"] += cds_scale * 0.10           # 650 bps -> +0.0065
    out["FX"]["sigma"] *= (1.0 + cds_scale * 1.6) # 650 bps -> ~+10% vol

    # sınırlar
    out["EQUITY"]["mu"] = clamp(out["EQUITY"]["mu"], -0.06, 0.06)
    out["TR_2Y_BOND"]["mu"] = clamp(out["TR_2Y_BOND"]["mu"], -0.03, 0.05)
    out["FX"]["mu"] = clamp(out["FX"]["mu"], -0.02, 0.08)

    return out

def simulate_round_returns(rng: np.random.Generator, weights: dict, dyn: dict, tr_price_effect: float) -> dict:
    rets = {}
    rets["CASH"] = 0.0

    rets["US_BOND"] = float(rng.normal(dyn["US_BOND"]["mu"], dyn["US_BOND"]["sigma"]))
    rets["EQUITY"]  = float(rng.normal(dyn["EQUITY"]["mu"], dyn["EQUITY"]["sigma"]))
    rets["FX"]      = float(rng.normal(dyn["FX"]["mu"], dyn["FX"]["sigma"]))

    # TR bond + faiz->fiyat etkisi
    tr_r = rng.normal(dyn["TR_2Y_BOND"]["mu"], dyn["TR_2Y_BOND"]["sigma"])
    tr_r += tr_price_effect
    rets["TR_2Y_BOND"] = float(tr_r)

    port_r = 0.0
    for a in ASSETS:
        port_r += weights[a] * rets[a]
    rets["PORTFOLIO"] = float(port_r)
    return rets

def benchmark_weights(name: str) -> dict:
    # FX eklendi: benchmark’lerde FX default 0, ama istersen aşağıya “Döviz Ağırlıklı” da ekledim.
    if name == "Hepsi Nakit":
        return {"TR_2Y_BOND": 0.0, "US_BOND": 0.0, "EQUITY": 0.0, "FX": 0.0, "CASH": 1.0}
    if name == "Hepsi TR Tahvil":
        return {"TR_2Y_BOND": 1.0, "US_BOND": 0.0, "EQUITY": 0.0, "FX": 0.0, "CASH": 0.0}
    if name == "60/40 (Borsa/TR Tahvil)":
        return {"TR_2Y_BOND": 0.40, "US_BOND": 0.0, "EQUITY": 0.60, "FX": 0.0, "CASH": 0.0}
    if name == "Dengeli (25/25/35/10/5)":
        return {"TR_2Y_BOND": 0.25, "US_BOND": 0.25, "EQUITY": 0.35, "FX": 0.10, "CASH": 0.05}
    if name == "Döviz Ağırlıklı (50 FX / 50 Nakit)":
        return {"TR_2Y_BOND": 0.0, "US_BOND": 0.0, "EQUITY": 0.0, "FX": 0.50, "CASH": 0.50}
    return {"TR_2Y_BOND": 0.25, "US_BOND": 0.25, "EQUITY": 0.35, "FX": 0.10, "CASH": 0.05}

# -----------------------------
# Session State
# -----------------------------
if "tur_idx" not in st.session_state:
    st.session_state.tur_idx = 0
if "capital" not in st.session_state:
    st.session_state.capital = float(STARTING_CAPITAL)
if "history" not in st.session_state:
    st.session_state.history = []
if "seed" not in st.session_state:
    st.session_state.seed = 42
if "prev_tr_yield" not in st.session_state:
    first = ROUNDS[0]
    st.session_state.prev_tr_yield = tr_yield_from_policy_cds_infl(first["policy"], first["cds"], first["inf_exp"])["yield"]

if "bench_capital" not in st.session_state:
    st.session_state.bench_capital = {
        "Hepsi Nakit": float(STARTING_CAPITAL),
        "Hepsi TR Tahvil": float(STARTING_CAPITAL),
        "60/40 (Borsa/TR Tahvil)": float(STARTING_CAPITAL),
        "Dengeli (25/25/35/10/5)": float(STARTING_CAPITAL),
        "Döviz Ağırlıklı (50 FX / 50 Nakit)": float(STARTING_CAPITAL),
    }

# Varsayılan yüzde kutuları (FX eklendi)
if "pct_tr" not in st.session_state:
    st.session_state.pct_tr = 35
if "pct_us" not in st.session_state:
    st.session_state.pct_us = 20
if "pct_eq" not in st.session_state:
    st.session_state.pct_eq = 30
if "pct_fx" not in st.session_state:
    st.session_state.pct_fx = 10
if "pct_cash" not in st.session_state:
    st.session_state.pct_cash = 5

# -----------------------------
# UI – Başlık + Reset
# -----------------------------
st.title("🎮 Risk mi Getiri mi? – Bireysel Portföy Oyunu (Kur Eklendi)")
st.caption("Rejim bazlı: Sakin / Enflasyon / Risk Şoku / Stres / İyileşme. Varlıklar: TR tahvil, US tahvil, borsa, USD/TRY, nakit.")

top_left, top_right = st.columns([1, 1])
with top_left:
    st.metric("Portföy Değeri", f"{st.session_state.capital:,.0f} TL")
with top_right:
    if st.button("🔄 Oyunu Sıfırla (Her şeyi başa al)"):
        st.session_state.tur_idx = 0
        st.session_state.capital = float(STARTING_CAPITAL)
        st.session_state.history = []
        st.session_state.seed = 42
        first = ROUNDS[0]
        st.session_state.prev_tr_yield = tr_yield_from_policy_cds_infl(first["policy"], first["cds"], first["inf_exp"])["yield"]
        st.session_state.bench_capital = {k: float(STARTING_CAPITAL) for k in st.session_state.bench_capital.keys()}
        st.session_state.pct_tr, st.session_state.pct_us, st.session_state.pct_eq, st.session_state.pct_fx, st.session_state.pct_cash = 35, 20, 30, 10, 5
        st.rerun()

st.markdown("---")
left, right = st.columns([1.2, 0.8])

# -----------------------------
# Sol: Bilgi kartları + tur kartı + mekanizma
# -----------------------------
with left:
    st.subheader("1) Varlık Kartları (Baz)")

    df_cards = pd.DataFrame([{
        "Varlık": ASSET_NAMES[a],
        "Baz Beklenen Getiri (tur)": f"{BASE[a]['mu']*100:.1f}%",
        "Baz Oynaklık (tur)": f"{BASE[a]['sigma']*100:.1f}%",
        "Not": (
            "Faiz ↑ → fiyat ↓ etkisi vardır" if a == "TR_2Y_BOND" else
            "Daha stabil" if a == "US_BOND" else
            "Daha oynak" if a == "EQUITY" else
            "CDS ↑ olunca kur baskısı artabilir" if a == "FX" else
            "Getiri yok"
        )
    } for a in ASSETS])

    st.dataframe(df_cards, use_container_width=True)

    st.subheader("2) Bu Tur Piyasa Kartı")

    if st.session_state.tur_idx >= N_ROUNDS:
        st.success("Oyun tamamlandı ✅ Aşağıdaki özet bölüme geç.")
    else:
        rinfo = ROUNDS[st.session_state.tur_idx]
        comps = tr_yield_from_policy_cds_infl(rinfo["policy"], rinfo["cds"], rinfo["inf_exp"])
        tr_y = comps["yield"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tur", f"{rinfo['tur']}/{N_ROUNDS}")
        c2.metric("Rejim", rinfo["rejim"])
        c3.metric("Politika", f"%{rinfo['policy']*100:.1f}")
        c4.metric("CDS", f"{rinfo['cds']} bps")
        c5.metric("Enflasyon bek.", f"%{rinfo['inf_exp']*100:.1f}")

        st.write(f"**Haber:** {rinfo['haber']}")
        st.metric("TR 2Y tahvil faizi (temsili)", f"%{tr_y*100:.1f}")

        dyn = get_dynamic_params(rinfo["rejim"], rinfo["cds"])
        st.write("**Bu tur (rejim + CDS) nedeniyle varlık profilleri (mu/sigma):**")
        df_dyn = pd.DataFrame([{
            "Varlık": ASSET_NAMES[a],
            "Beklenen getiri (tur)": f"{dyn[a]['mu']*100:.2f}%",
            "Oynaklık (tur)": f"{dyn[a]['sigma']*100:.2f}%"
        } for a in ASSETS])
        st.dataframe(df_dyn, use_container_width=True)

        with st.expander("Mekanizmayı Aç (CDS → faiz → tahvil fiyatı)", expanded=True):
            st.write("**TR Tahvil Faizi Bileşenleri (temsili):**")
            df_comp = pd.DataFrame([
                {"Bileşen": "Politika faizi", "Katkı": comps["policy"]},
                {"Bileşen": "Enflasyon bekl. katmanı", "Katkı": comps["infl_layer"]},
                {"Bileşen": "Risk primi (CDS etkisi)", "Katkı": comps["risk_premium"]},
                {"Bileşen": "Toplam TR 2Y faiz", "Katkı": comps["yield"]},
            ])
            df_comp["Katkı (%)"] = (df_comp["Katkı"] * 100).round(2).astype(str) + "%"
            st.dataframe(df_comp[["Bileşen", "Katkı (%)"]], use_container_width=True)

            prev_y = st.session_state.prev_tr_yield
            prev_p = bond_price_from_yield(prev_y)
            curr_p = bond_price_from_yield(tr_y)
            price_effect = (curr_p - prev_p) / prev_p

            st.write("**Tahvil Fiyatı – Faiz Ters İlişkisi (temsili):**")
            st.write(f"- Önceki faiz: **%{prev_y*100:.1f}** → fiyat endeksi: **{prev_p:.2f}**")
            st.write(f"- Bu tur faiz: **%{tr_y*100:.1f}** → fiyat endeksi: **{curr_p:.2f}**")
            st.write(f"- Faiz değişiminin fiyat etkisi (yaklaşık): **{price_effect*100:.2f}%**")

            st.info("Özet: CDS ↑ çoğu zaman risk primi ↑ demektir → tahvil faizi ↑. Faiz ↑ olunca tahvil fiyatı düşme eğilimindedir. CDS ↑ aynı zamanda kur tarafında baskı yaratabilir.")

# -----------------------------
# Sağ: Karar + Sonuç
# -----------------------------
with right:
    st.subheader("3) Karar Ver ve Oyna")

    if st.session_state.tur_idx < N_ROUNDS:
        rinfo = ROUNDS[st.session_state.tur_idx]
        comps = tr_yield_from_policy_cds_infl(rinfo["policy"], rinfo["cds"], rinfo["inf_exp"])
        tr_y = comps["yield"]

        st.write("Yüzdeleri kutucuklardan gir (toplam %100):")

        i1, i2 = st.columns(2)
        with i1:
            pct_tr = st.number_input("Türkiye 2Y Tahvil (%)", min_value=0, max_value=100, value=int(st.session_state.pct_tr), step=1, key="pct_tr")
            pct_eq = st.number_input("Borsa Endeksi (%)", min_value=0, max_value=100, value=int(st.session_state.pct_eq), step=1, key="pct_eq")
            pct_cash = st.number_input("Nakit (%)", min_value=0, max_value=100, value=int(st.session_state.pct_cash), step=1, key="pct_cash")
        with i2:
            pct_us = st.number_input("ABD Tahvil (%)", min_value=0, max_value=100, value=int(st.session_state.pct_us), step=1, key="pct_us")
            pct_fx = st.number_input("USD/TRY (Kur) (%)", min_value=0, max_value=100, value=int(st.session_state.pct_fx), step=1, key="pct_fx")

        pcts = {"TR_2Y_BOND": pct_tr, "US_BOND": pct_us, "EQUITY": pct_eq, "FX": pct_fx, "CASH": pct_cash}
        ok, msg, total = validate_percentages(pcts)

        st.write(f"Toplam: **%{total}**")
        if not ok:
            st.error(msg)

        weights = {a: pcts[a] / 100.0 for a in ASSETS}

        # Bu tur için dinamik mu/sigma ile "beklenen sonuç" göster
        dyn = get_dynamic_params(rinfo["rejim"], rinfo["cds"])
        means = {a: dyn[a]["mu"] for a in ASSETS}
        sigmas = {a: dyn[a]["sigma"] for a in ASSETS}
        exp_mu, exp_sigma = portfolio_stats(weights, means, sigmas)

        m1, m2 = st.columns(2)
        m1.metric("Beklenen getiri (bu tur)", f"{exp_mu*100:.2f}%")
        m2.metric("Tahmini risk (bu tur)", f"{exp_sigma*100:.2f}%")

        run_btn = st.button("▶️ Turu Oyna", disabled=(not ok))

        if run_btn and ok:
            # Tahvil fiyat etkisi
            prev_y = st.session_state.prev_tr_yield
            prev_p = bond_price_from_yield(prev_y)
            curr_p = bond_price_from_yield(tr_y)
            tr_price_effect = (curr_p - prev_p) / prev_p

            rng = np.random.default_rng(st.session_state.seed + rinfo["tur"] * 101)

            asset_rets = simulate_round_returns(rng, weights, dyn, tr_price_effect)
            port_r = asset_rets["PORTFOLIO"]

            old_cap = st.session_state.capital
            new_cap = old_cap * (1.0 + port_r)
            st.session_state.capital = float(new_cap)

            # Benchmark’ler
            for bname in list(st.session_state.bench_capital.keys()):
                bw = benchmark_weights(bname)
                b_rets = simulate_round_returns(rng, bw, dyn, tr_price_effect)
                st.session_state.bench_capital[bname] *= (1.0 + b_rets["PORTFOLIO"])

            # Log
            st.session_state.history.append({
                "Tur": rinfo["tur"],
                "Rejim": rinfo["rejim"],
                "Haber": rinfo["haber"],
                "Politika": rinfo["policy"],
                "CDS": rinfo["cds"],
                "EnflasyonBek": rinfo["inf_exp"],
                "TR_Tahvil_Faiz": tr_y,
                "Tahvil_Fiyat_Etkisi": tr_price_effect,
                "TR_Tahvil_Getiri": asset_rets["TR_2Y_BOND"],
                "US_Tahvil_Getiri": asset_rets["US_BOND"],
                "Borsa_Getiri": asset_rets["EQUITY"],
                "Kur_Getiri": asset_rets["FX"],
                "Nakit_Getiri": asset_rets["CASH"],
                "Portföy_Getiri": port_r,
                "Portföy_Değeri": new_cap,
                "Ağırlık_TR": weights["TR_2Y_BOND"],
                "Ağırlık_US": weights["US_BOND"],
                "Ağırlık_Borsa": weights["EQUITY"],
                "Ağırlık_Kur": weights["FX"],
                "Ağırlık_Nakit": weights["CASH"],
            })

            st.markdown("### Tur Sonucu")
            s1, s2 = st.columns(2)
            s1.metric("Portföy getirisi", f"{port_r*100:.2f}%")
            s2.metric("Yeni portföy değeri", f"{new_cap:,.0f} TL")

            df_ret = pd.DataFrame([{
                "Varlık": ASSET_NAMES[a],
                "Ağırlık": f"{weights[a]*100:.0f}%",
                "Getiri (tur)": f"{asset_rets[a]*100:.2f}%"
            } for a in ASSETS])
            st.dataframe(df_ret, use_container_width=True)

            st.info("Tartışma: CDS yükseldiğinde borsa ve kur davranışı nasıl değişti? Tahvil faizindeki artış tahvil fiyatına nasıl yansıdı?")

            # ilerle
            st.session_state.prev_tr_yield = tr_y
            st.session_state.tur_idx += 1
            st.rerun()
    else:
        st.success("Oyun tamamlandı ✅")

# -----------------------------
# Özet + Benchmark
# -----------------------------
st.markdown("---")
st.subheader("📊 Özet (Tur Tur) + Kıyas")

if len(st.session_state.history) == 0:
    st.write("Henüz tur oynanmadı.")
else:
    df_hist = pd.DataFrame(st.session_state.history)

    show_cols = [
        "Tur", "Rejim", "CDS", "TR_Tahvil_Faiz", "Tahvil_Fiyat_Etkisi",
        "Portföy_Getiri", "Portföy_Değeri",
        "Ağırlık_TR", "Ağırlık_US", "Ağırlık_Borsa", "Ağırlık_Kur", "Ağırlık_Nakit"
    ]
    df_show = df_hist[show_cols].copy()

    df_show["TR_Tahvil_Faiz"] = df_show["TR_Tahvil_Faiz"] * 100
    df_show["Tahvil_Fiyat_Etkisi"] = df_show["Tahvil_Fiyat_Etkisi"] * 100
    df_show["Portföy_Getiri"] = df_show["Portföy_Getiri"] * 100
    for c in ["Ağırlık_TR", "Ağırlık_US", "Ağırlık_Borsa", "Ağırlık_Kur", "Ağırlık_Nakit"]:
        df_show[c] = df_show[c] * 100

    st.dataframe(
        df_show.style.format({
            "TR_Tahvil_Faiz": "{:.1f}%",
            "Tahvil_Fiyat_Etkisi": "{:.2f}%",
            "Portföy_Getiri": "{:.2f}%",
            "Portföy_Değeri": "{:,.0f}",
            "Ağırlık_TR": "{:.0f}%",
            "Ağırlık_US": "{:.0f}%",
            "Ağırlık_Borsa": "{:.0f}%",
            "Ağırlık_Kur": "{:.0f}%",
            "Ağırlık_Nakit": "{:.0f}%",
        }),
        use_container_width=True
    )

    st.markdown("### Kıyas (Benchmark Stratejiler)")
    bench_df = pd.DataFrame([{"Strateji": k, "Değer": v} for k, v in st.session_state.bench_capital.items()]).sort_values("Değer", ascending=False)
    bench_df = pd.concat([pd.DataFrame([{"Strateji": "Senin Portföyün", "Değer": st.session_state.capital}]), bench_df], ignore_index=True)
    st.dataframe(bench_df.style.format({"Değer": "{:,.0f}"}), use_container_width=True)

    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Sonuçları CSV indir", data=csv, file_name="bireysel_portfoy_oyunu_sonuclar.csv", mime="text/csv")
