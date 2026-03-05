# app.py
# Borsa Uygulamaları – Bireysel Portföy Oyunu
# CDS – Politika faizi – Enflasyon beklentisi → TR tahvil faizi
# Faiz ↑ → Tahvil fiyatı ↓ (fiyat etkisi)
#
# Değişiklikler:
# - "Karar ver ve oyna" bölümünde slider yerine % kutucuk (number_input) var.
# - Mini soru kaldırıldı.
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

ASSETS = ["TR_2Y_BOND", "US_BOND", "EQUITY", "CASH"]
ASSET_NAMES = {
    "TR_2Y_BOND": "Türkiye 2Y Tahvil",
    "US_BOND": "ABD Tahvil (Düşük risk)",
    "EQUITY": "Borsa Endeksi (Riskli)",
    "CASH": "Nakit",
}

# Normal koşullarda (tur başına) temsili getiri/oynaklık
NORMAL_PARAMS = {
    "TR_2Y_BOND": {"mu": 0.020, "sigma": 0.015},
    "US_BOND":    {"mu": 0.008, "sigma": 0.005},
    "EQUITY":     {"mu": 0.030, "sigma": 0.060},
    "CASH":       {"mu": 0.000, "sigma": 0.000},
}

ROUNDS = [
    {"tur": 1, "haber": "Sakin dönem: risk algısı düşük",                          "policy": 0.35, "cds": 250, "inf_exp": 0.30},
    {"tur": 2, "haber": "Enflasyon beklentisi yükseliyor",                         "policy": 0.35, "cds": 320, "inf_exp": 0.45},
    {"tur": 3, "haber": "Risk algısı bozuluyor: CDS yükseldi",                     "policy": 0.40, "cds": 650, "inf_exp": 0.50},
    {"tur": 4, "haber": "Belirsizlik sürüyor: risk primi yüksek",                  "policy": 0.45, "cds": 800, "inf_exp": 0.55},
    {"tur": 5, "haber": "Kısmi iyileşme: CDS düşüyor, beklentiler toparlanıyor",   "policy": 0.40, "cds": 420, "inf_exp": 0.40},
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
        return False, "Toplam **%100** olmalı. (İpucu: kutulardan birini azaltıp/artırın.)", total
    if any(v < 0 for v in pcts.values()):
        return False, "Yüzdeler negatif olamaz.", total
    return True, "", total

def portfolio_stats(weights: dict, means: dict, sigmas: dict) -> tuple[float, float]:
    mu_p = sum(weights[a] * means[a] for a in ASSETS)
    var_p = sum((weights[a] ** 2) * (sigmas[a] ** 2) for a in ASSETS)  # korelasyon yok varsayımı
    return float(mu_p), float(np.sqrt(var_p))

def simulate_round_returns(rng: np.random.Generator, weights: dict, tr_price_effect: float, cds_bps: int) -> dict:
    rets = {}

    rets["US_BOND"] = float(rng.normal(NORMAL_PARAMS["US_BOND"]["mu"], NORMAL_PARAMS["US_BOND"]["sigma"]))
    rets["CASH"] = 0.0

    cds_penalty = (cds_bps / 10000.0) * 0.06  # temsili
    eq_mu = clamp(NORMAL_PARAMS["EQUITY"]["mu"] - cds_penalty, -0.05, 0.05)
    rets["EQUITY"] = float(rng.normal(eq_mu, NORMAL_PARAMS["EQUITY"]["sigma"]))

    rets["TR_2Y_BOND"] = float(rng.normal(NORMAL_PARAMS["TR_2Y_BOND"]["mu"], NORMAL_PARAMS["TR_2Y_BOND"]["sigma"]) + tr_price_effect)

    port_r = 0.0
    for a in ASSETS:
        port_r += weights[a] * rets[a]
    rets["PORTFOLIO"] = float(port_r)
    return rets

def benchmark_weights(name: str) -> dict:
    if name == "Hepsi Nakit":
        return {"TR_2Y_BOND": 0.0, "US_BOND": 0.0, "EQUITY": 0.0, "CASH": 1.0}
    if name == "Hepsi TR Tahvil":
        return {"TR_2Y_BOND": 1.0, "US_BOND": 0.0, "EQUITY": 0.0, "CASH": 0.0}
    if name == "60/40 (Borsa/TR Tahvil)":
        return {"TR_2Y_BOND": 0.40, "US_BOND": 0.0, "EQUITY": 0.60, "CASH": 0.0}
    if name == "Dengeli (25/25/40/10)":
        return {"TR_2Y_BOND": 0.25, "US_BOND": 0.25, "EQUITY": 0.40, "CASH": 0.10}
    return {"TR_2Y_BOND": 0.25, "US_BOND": 0.25, "EQUITY": 0.40, "CASH": 0.10}

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
        "Dengeli (25/25/40/10)": float(STARTING_CAPITAL),
    }

# Varsayılan yüzde kutuları için state
if "pct_tr" not in st.session_state:
    st.session_state.pct_tr = 40
if "pct_us" not in st.session_state:
    st.session_state.pct_us = 20
if "pct_eq" not in st.session_state:
    st.session_state.pct_eq = 30
if "pct_cash" not in st.session_state:
    st.session_state.pct_cash = 10

# -----------------------------
# UI – Başlık + Reset
# -----------------------------
st.title("🎮 Risk mi Getiri mi? – Bireysel Portföy Oyunu")
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
        st.session_state.pct_tr, st.session_state.pct_us, st.session_state.pct_eq, st.session_state.pct_cash = 40, 20, 30, 10
        st.rerun()

st.markdown("---")

left, right = st.columns([1.15, 0.85])

# -----------------------------
# Sol: Varlık kartları + tur kartı + mekanizma
# -----------------------------
with left:
    st.subheader("1) Varlık Kartları (Temel Getiri/Risk Bilgisi)")

    df_cards = pd.DataFrame([{
        "Varlık": ASSET_NAMES[a],
        "Beklenen Getiri (tur)": f"{NORMAL_PARAMS[a]['mu']*100:.1f}%",
        "Oynaklık (tur)": f"{NORMAL_PARAMS[a]['sigma']*100:.1f}%",
        "Kısa Not": (
            "CDS ↑ olursa faiz ↑; ama faiz ↑ → fiyat ↓ olabilir" if a == "TR_2Y_BOND" else
            "Daha stabil, düşük getiri" if a == "US_BOND" else
            "Yüksek oynaklık, CDS ↑ olunca baskı artabilir" if a == "EQUITY" else
            "Risk yok, getiri yok"
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tur", f"{rinfo['tur']}/{N_ROUNDS}")
        c2.metric("Politika faizi", f"%{rinfo['policy']*100:.1f}")
        c3.metric("CDS", f"{rinfo['cds']} bps")
        c4.metric("Enflasyon bekl.", f"%{rinfo['inf_exp']*100:.1f}")

        st.write(f"**Haber:** {rinfo['haber']}")
        st.metric("TR 2Y tahvil faizi (temsili)", f"%{tr_y*100:.1f}")

        with st.expander("Mekanizmayı Aç (CDS → faiz → fiyat)", expanded=True):
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
            st.info("Özet: **Faiz yükseldikçe tahvil fiyatı düşme eğilimindedir**. Bu yüzden yüksek faiz her zaman “kazanç” demek değildir.")

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
        with i2:
            pct_us = st.number_input("ABD Tahvil (%)", min_value=0, max_value=100, value=int(st.session_state.pct_us), step=1, key="pct_us")
            pct_cash = st.number_input("Nakit (%)", min_value=0, max_value=100, value=int(st.session_state.pct_cash), step=1, key="pct_cash")

        pcts = {"TR_2Y_BOND": pct_tr, "US_BOND": pct_us, "EQUITY": pct_eq, "CASH": pct_cash}
        ok, msg, total = validate_percentages(pcts)

        st.write(f"Toplam: **%{total}**")
        if not ok:
            st.error(msg)

        weights = {a: pcts[a] / 100.0 for a in ASSETS}

        # Beklenen sonuç paneli (kararı kolaylaştırır)
        means = {a: NORMAL_PARAMS[a]["mu"] for a in ASSETS}
        sigmas = {a: NORMAL_PARAMS[a]["sigma"] for a in ASSETS}
        exp_mu, exp_sigma = portfolio_stats(weights, means, sigmas)

        p1, p2 = st.columns(2)
        p1.metric("Beklenen getiri (tur)", f"{exp_mu*100:.2f}%")
        p2.metric("Tahmini risk (oynaklık)", f"{exp_sigma*100:.2f}%")

        run_btn = st.button("▶️ Turu Oyna", disabled=(not ok))

        if run_btn and ok:
            # Tahvil fiyat etkisi (faiz değişimi)
            prev_y = st.session_state.prev_tr_yield
            prev_p = bond_price_from_yield(prev_y)
            curr_p = bond_price_from_yield(tr_y)
            tr_price_effect = (curr_p - prev_p) / prev_p

            rng = np.random.default_rng(st.session_state.seed + rinfo["tur"] * 101)

            asset_rets = simulate_round_returns(rng, weights, tr_price_effect, rinfo["cds"])
            port_r = asset_rets["PORTFOLIO"]

            old_cap = st.session_state.capital
            new_cap = old_cap * (1.0 + port_r)
            st.session_state.capital = float(new_cap)

            # Benchmark’ler
            for bname in list(st.session_state.bench_capital.keys()):
                bw = benchmark_weights(bname)
                b_rets = simulate_round_returns(rng, bw, tr_price_effect, rinfo["cds"])
                st.session_state.bench_capital[bname] *= (1.0 + b_rets["PORTFOLIO"])

            st.session_state.history.append({
                "Tur": rinfo["tur"],
                "Haber": rinfo["haber"],
                "Politika": rinfo["policy"],
                "CDS": rinfo["cds"],
                "EnflasyonBek": rinfo["inf_exp"],
                "TR_Tahvil_Faiz": tr_y,
                "Tahvil_Fiyat_Etkisi": tr_price_effect,
                "TR_Tahvil_Getiri": asset_rets["TR_2Y_BOND"],
                "US_Tahvil_Getiri": asset_rets["US_BOND"],
                "Borsa_Getiri": asset_rets["EQUITY"],
                "Nakit_Getiri": asset_rets["CASH"],
                "Portföy_Getiri": port_r,
                "Portföy_Değeri": new_cap,
                "Ağırlık_TR": weights["TR_2Y_BOND"],
                "Ağırlık_US": weights["US_BOND"],
                "Ağırlık_Borsa": weights["EQUITY"],
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

            st.info("Tartışma: TR tahvil faizi yükseldi mi? Tahvil fiyat etkisi (faiz ↑ → fiyat ↓) portföyünü nasıl etkiledi?")

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
        "Tur", "CDS", "TR_Tahvil_Faiz", "Tahvil_Fiyat_Etkisi",
        "Portföy_Getiri", "Portföy_Değeri",
        "Ağırlık_TR", "Ağırlık_US", "Ağırlık_Borsa", "Ağırlık_Nakit"
    ]
    df_show = df_hist[show_cols].copy()

    df_show["TR_Tahvil_Faiz"] = df_show["TR_Tahvil_Faiz"] * 100
    df_show["Tahvil_Fiyat_Etkisi"] = df_show["Tahvil_Fiyat_Etkisi"] * 100
    df_show["Portföy_Getiri"] = df_show["Portföy_Getiri"] * 100
    for c in ["Ağırlık_TR", "Ağırlık_US", "Ağırlık_Borsa", "Ağırlık_Nakit"]:
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
