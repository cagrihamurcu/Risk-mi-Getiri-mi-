# app.py
# Borsa Uygulamaları – Bireysel Portföy Oyunu
# CDS – Faiz – Tahvil Fiyatı – Kuyruk Riski (Tail Risk) + Basit VaR Bilgi Kutusu
#
# Çalıştır:
#   pip install streamlit numpy pandas
#   streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Risk mi Getiri mi? | Bireysel Portföy Oyunu", layout="wide")

# -----------------------------
# OYUN AYARLARI (sade ve anlaşılır)
# -----------------------------
N_ROUNDS = 5
STARTING_CAPITAL = 100_000

# Varlıklar
ASSETS = ["TR_2Y_BOND", "US_BOND", "EQUITY", "CASH"]
ASSET_NAMES = {
    "TR_2Y_BOND": "Türkiye 2Y Tahvil",
    "US_BOND": "ABD Tahvil (Düşük risk)",
    "EQUITY": "Borsa Endeksi (Riskli)",
    "CASH": "Nakit",
}

# "Normal" dönem temsili parametreleri (tur başına)
# Not: Öğretim amaçlı basitleştirme. Amaç mekanizma.
NORMAL_PARAMS = {
    "TR_2Y_BOND": {"mu": 0.020, "sigma": 0.015},
    "US_BOND":    {"mu": 0.008, "sigma": 0.005},
    "EQUITY":     {"mu": 0.030, "sigma": 0.060},
    "CASH":       {"mu": 0.000, "sigma": 0.000},
}

# Kuyruk olayında (tail event) varlık şokları (temsili)
TAIL_SHOCK = {
    "TR_2Y_BOND": {"mu": -0.080, "sigma": 0.030},   # -%5 ile -%12 civarı olasılık
    "US_BOND":    {"mu": -0.005, "sigma": 0.010},   # sınırlı etkilenme
    "EQUITY":     {"mu": -0.220, "sigma": 0.060},   # -%15 ile -%30+ civarı olasılık
    "CASH":       {"mu":  0.000, "sigma": 0.000},
}

# Senaryo turları: Politika faizi, CDS (bps), enflasyon beklentisi
# (Slayt mantığı: CDS ↑ → risk primi ↑ → tahvil faizi ↑ ; faiz ↑ → tahvil fiyatı ↓)
ROUNDS = [
    {"tur": 1, "haber": "Sakin dönem: risk algısı düşük",              "policy": 0.35, "cds": 250, "inf_exp": 0.30},
    {"tur": 2, "haber": "Enflasyon beklentisi yükseliyor",             "policy": 0.35, "cds": 320, "inf_exp": 0.45},
    {"tur": 3, "haber": "Risk şoku: CDS hızla yükseldi",               "policy": 0.40, "cds": 650, "inf_exp": 0.50},
    {"tur": 4, "haber": "Küresel sıkılaşma + belirsizlik sürüyor",     "policy": 0.45, "cds": 800, "inf_exp": 0.55},
    {"tur": 5, "haber": "Kısmi iyileşme: CDS düşüyor, beklentiler toparlanıyor", "policy": 0.40, "cds": 420, "inf_exp": 0.40},
]

# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def tail_probability_from_cds(cds_bps: int) -> float:
    """
    CDS arttıkça kuyruk olayı olasılığı artsın (temsili).
    200 bps ~ %2 ; 1000 bps ~ %12 civarı
    """
    p = 0.01 + (cds_bps / 10000.0) * 0.12
    return clamp(p, 0.02, 0.15)

def tr_yield_from_policy_cds_infl(policy: float, cds_bps: int, inf_exp: float) -> float:
    """
    TR 2Y tahvil faizi (temsili):
      y = politika + (enflasyon beklentisi katmanı) + (risk primi)
    """
    k = 1.10
    risk_premium = (cds_bps / 10000.0) * k
    inflation_layer = 0.30 * inf_exp
    y = policy + inflation_layer + risk_premium
    return max(y, 0.0)

def bond_price_from_yield(y: float, duration: float = 1.8, face: float = 100.0) -> float:
    """Basit fiyat yaklaşımı: Fiyat ≈ Face / (1+y)^duration"""
    y = max(y, 1e-6)
    return face / ((1.0 + y) ** duration)

def simulate_asset_returns(rng: np.random.Generator, tail_event: bool, tr_bond_price_effect: float) -> dict:
    """
    Bu tur varlık getirilerini üretir.
    TR tahvil: normal getiri + (faiz değişiminden fiyat etkisi) + (tail şoku varsa ek şok)
    """
    rets = {}

    # US bond, equity, cash: direkt normal + tail ek şok
    for a in ["US_BOND", "EQUITY", "CASH"]:
        mu = NORMAL_PARAMS[a]["mu"]
        sigma = NORMAL_PARAMS[a]["sigma"]
        r = rng.normal(mu, sigma)

        if tail_event:
            r += rng.normal(TAIL_SHOCK[a]["mu"], TAIL_SHOCK[a]["sigma"])

        rets[a] = float(r)

    # TR bond: normal + fiyat etkisi + tail ek şok
    a = "TR_2Y_BOND"
    r_tr = rng.normal(NORMAL_PARAMS[a]["mu"], NORMAL_PARAMS[a]["sigma"])
    r_tr += tr_bond_price_effect  # faiz ↑ → fiyat ↓ (genelde negatif etki)
    if tail_event:
        r_tr += rng.normal(TAIL_SHOCK[a]["mu"], TAIL_SHOCK[a]["sigma"])
    rets[a] = float(r_tr)

    return rets

def portfolio_return(weights: dict, asset_returns: dict) -> float:
    r = 0.0
    for a, w in weights.items():
        r += w * asset_returns[a]
    return float(r)

def historical_var_95(simulated_port_returns: np.ndarray) -> float:
    """VaR(95): en kötü %5 eşiği (kayıp olarak pozitif raporlanır)."""
    q = np.quantile(simulated_port_returns, 0.05)
    return float(max(0.0, -q))

def validate_weights(w: dict) -> tuple[bool, str]:
    s = sum(w.values())
    if abs(s - 1.0) > 1e-9:
        return False, "Ağırlıkların toplamı %100 olmalı."
    for k, v in w.items():
        if v < 0:
            return False, "Ağırlıklar negatif olamaz."
    return True, ""

# -----------------------------
# Session state
# -----------------------------
if "seed" not in st.session_state:
    st.session_state.seed = 42

if "tur_idx" not in st.session_state:
    st.session_state.tur_idx = 0

if "capital" not in st.session_state:
    st.session_state.capital = STARTING_CAPITAL

if "history" not in st.session_state:
    st.session_state.history = []  # her tur sonuç kaydı

if "prev_tr_yield" not in st.session_state:
    # ilk turda "önceki" yok; aynı kabul edelim
    first = ROUNDS[0]
    st.session_state.prev_tr_yield = tr_yield_from_policy_cds_infl(first["policy"], first["cds"], first["inf_exp"])

# -----------------------------
# Üst başlık
# -----------------------------
st.title("🎮 Risk mi Getiri mi? – Bireysel Portföy Oyunu (CDS / Faiz / Tahvil / Kuyruk Riski)")
st.caption("Bu uygulama öğretim amaçlı temsili bir simülasyondur. Amaç mekanizmayı netleştirmektir.")

# -----------------------------
# Sidebar: Kontroller
# -----------------------------
with st.sidebar:
    st.subheader("Kontroller")
    st.number_input("Rastgelelik tohumu (seed)", min_value=0, max_value=999999, value=st.session_state.seed, key="seed")

    st.write("---")
    if st.button("🔄 Oyunu Sıfırla"):
        st.session_state.tur_idx = 0
        st.session_state.capital = STARTING_CAPITAL
        st.session_state.history = []
        first = ROUNDS[0]
        st.session_state.prev_tr_yield = tr_yield_from_policy_cds_infl(first["policy"], first["cds"], first["inf_exp"])
        st.rerun()

    st.write("---")
    st.subheader("Hızlı Hatırlatma")
    st.write("- **CDS ↑ → risk primi ↑ → tahvil faizi ↑**")
    st.write("- **Faiz ↑ → tahvil fiyatı ↓** (eski tahvil yeniden fiyatlanır)")
    st.write("- **Kuyruk riski:** nadir ama yıkıcı kayıp")
    st.write("- **VaR(95):** en kötü %5 eşiği, *en uç felaketi garanti etmez*")

# -----------------------------
# Ana düzen: 3 sütun
# -----------------------------
col1, col2, col3 = st.columns([1.05, 1.10, 0.95])

# 1) Varlık Kartları
with col1:
    st.subheader("1) Varlık Kartları (Temel Bilgi)")

    card_rows = []
    for a in ASSETS:
        card_rows.append({
            "Varlık": ASSET_NAMES[a],
            "Normal Beklenen Getiri (tur)": f"{NORMAL_PARAMS[a]['mu']*100:.1f}%",
            "Normal Oynaklık": f"{NORMAL_PARAMS[a]['sigma']*100:.1f}%",
            "Kuyruk Olayında Tipik Etki": (
                "TR tahvil: faiz ↑ olabilir ama fiyat düşebilir; ayrıca şok zarar yazabilir" if a == "TR_2Y_BOND" else
                "Sınırlı negatif etki" if a == "US_BOND" else
                "Sert düşüş riski (-%15 ila -%30+)" if a == "EQUITY" else
                "Etkilenmez"
            )
        })

    df_cards = pd.DataFrame(card_rows)
    st.dataframe(df_cards, use_container_width=True)

    st.info(
        "Önemli: Bu değerler **temsili**. Oyunda amaç; CDS yükselince risk algısının artmasını, "
        "tahvil faizinin yükselmesini ve **tahvil fiyatının düşebilmesini** deneyimlemektir."
    )

# 2) Piyasa Haber Kartı
with col2:
    st.subheader("2) Piyasa Haber Kartı (Bu Tur)")

    tur_idx = st.session_state.tur_idx
    if tur_idx >= N_ROUNDS:
        st.success("Oyun tamamlandı. Aşağıda özet sonuçları görebilirsiniz.")
    else:
        rinfo = ROUNDS[tur_idx]
        policy = rinfo["policy"]
        cds = rinfo["cds"]
        inf_exp = rinfo["inf_exp"]

        tr_y = tr_yield_from_policy_cds_infl(policy, cds, inf_exp)
        tail_p = tail_probability_from_cds(cds)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tur", f"{rinfo['tur']}/{N_ROUNDS}")
        k2.metric("Politika faizi", f"%{policy*100:.1f}")
        k3.metric("CDS", f"{cds} bps")
        k4.metric("Enflasyon bekl.", f"%{inf_exp*100:.1f}")

        st.markdown(f"**Haber:** {rinfo['haber']}")
        st.write("")

        m1, m2, m3 = st.columns(3)
        m1.metric("TR 2Y tahvil faizi (temsili)", f"%{tr_y*100:.1f}")
        m2.metric("Kuyruk olayı olasılığı", f"%{tail_p*100:.1f}")
        m3.metric("Mevcut Portföy Değeri", f"{st.session_state.capital:,.0f} TL")

        st.warning("Hatırlatma: **Faiz yükselirse** (özellikle hızlı) **tahvil fiyatı düşebilir**.")

# 3) Karar + Sonuç
with col3:
    st.subheader("3) Karar (Portföy)")

    if st.session_state.tur_idx < N_ROUNDS:
        # Ağırlık girişleri (tek ekranda, anlaşılır)
        st.write("Portföyünü dağıt (toplam %100):")

        w_tr = st.slider("Türkiye 2Y Tahvil (%)", 0, 100, 40)
        w_us = st.slider("ABD Tahvil (%)", 0, 100, 20)
        w_eq = st.slider("Borsa Endeksi (%)", 0, 100, 30)
        w_cash = st.slider("Nakit (%)", 0, 100, 10)

        total = w_tr + w_us + w_eq + w_cash
        st.write(f"Toplam: **%{total}**")

        weights = {
            "TR_2Y_BOND": w_tr / 100.0,
            "US_BOND": w_us / 100.0,
            "EQUITY": w_eq / 100.0,
            "CASH": w_cash / 100.0,
        }

        ok, msg = validate_weights(weights)
        if not ok:
            st.error(msg)

        # Oynat
        run_btn = st.button("▶️ Turu Oyna", disabled=(not ok))

        if run_btn and ok:
            rinfo = ROUNDS[st.session_state.tur_idx]
            policy = rinfo["policy"]
            cds = rinfo["cds"]
            inf_exp = rinfo["inf_exp"]

            # Bu tur faiz + fiyat etkisi
            curr_tr_yield = tr_yield_from_policy_cds_infl(policy, cds, inf_exp)
            prev_tr_yield = st.session_state.prev_tr_yield

            prev_price = bond_price_from_yield(prev_tr_yield)
            curr_price = bond_price_from_yield(curr_tr_yield)
            tr_price_effect = (curr_price - prev_price) / prev_price  # faiz ↑ ise genelde negatif

            # Kuyruk olayı?
            rng = np.random.default_rng(st.session_state.seed + rinfo["tur"] * 101)
            tail_event = (rng.random() < tail_probability_from_cds(cds))

            # Varlık getirileri
            asset_rets = simulate_asset_returns(rng, tail_event, tr_price_effect)

            # Portföy getirisi
            port_r = portfolio_return(weights, asset_rets)
            old_cap = st.session_state.capital
            new_cap = old_cap * (1.0 + port_r)
            st.session_state.capital = float(new_cap)

            # Basit VaR bilgi kutusu için 300 simülasyon
            sims = []
            for _ in range(300):
                te = (rng.random() < tail_probability_from_cds(cds))
                # aynı tur için tekrar tekrar getiriler üret
                # (fiyat etkisini sabit tutuyoruz; amaç eğitimsel basitlik)
                sim_rets = simulate_asset_returns(rng, te, tr_price_effect)
                sims.append(portfolio_return(weights, sim_rets))
            var95 = historical_var_95(np.array(sims))

            # Kaydet
            st.session_state.history.append({
                "Tur": rinfo["tur"],
                "Haber": rinfo["haber"],
                "Politika": policy,
                "CDS": cds,
                "EnflasyonBek": inf_exp,
                "TR_Tahvil_Faiz": curr_tr_yield,
                "KuyrukOlay": "EVET" if tail_event else "HAYIR",
                "TR_Tahvil_Getiri": asset_rets["TR_2Y_BOND"],
                "US_Tahvil_Getiri": asset_rets["US_BOND"],
                "Borsa_Getiri": asset_rets["EQUITY"],
                "Nakit_Getiri": asset_rets["CASH"],
                "Portföy_Getiri": port_r,
                "VaR95": var95,
                "Portföy_Değeri": new_cap,
                "Ağırlık_TR": weights["TR_2Y_BOND"],
                "Ağırlık_US": weights["US_BOND"],
                "Ağırlık_Borsa": weights["EQUITY"],
                "Ağırlık_Nakit": weights["CASH"],
            })

            # Sonuç ekranı
            if tail_event:
                st.error("⚠️ Bu tur **kuyruk olayı** gerçekleşti! (Nadir ama yıkıcı şok)")
            else:
                st.success("Bu tur normal koşullarda gerçekleşti.")

            st.markdown("### Tur Sonucu")
            s1, s2, s3 = st.columns(3)
            s1.metric("Portföy getirisi", f"{port_r*100:.2f}%")
            s2.metric("Yeni portföy değeri", f"{new_cap:,.0f} TL")
            s3.metric("VaR(95) (bilgi)", f"{var95*100:.2f}%")

            st.caption("VaR(95) en kötü %5 eşiği gösterir; **en uç felaketi garanti etmez**.")

            # Varlık bazında göster
            df_ret = pd.DataFrame([{
                "Varlık": ASSET_NAMES[a],
                "Ağırlık": f"{weights[a]*100:.0f}%",
                "Getiri (tur)": f"{asset_rets[a]*100:.2f}%"
            } for a in ASSETS])

            st.dataframe(df_ret, use_container_width=True)

            # Sonraki tura geçmeden önce: önceki TR yield güncelle
            st.session_state.prev_tr_yield = curr_tr_yield
            st.session_state.tur_idx += 1

            st.info(
                "Tartışma: CDS yükselince tahvil faizi neden yükseldi? "
                "Peki faiz yükselmesine rağmen tahvil neden zarar yazmış olabilir? (Faiz ↑ → fiyat ↓)"
            )

            st.rerun()
    else:
        st.success("Oyun bitti ✅ Aşağıdaki özet tabloya bakın.")

# -----------------------------
# OYUN ÖZETİ
# -----------------------------
st.markdown("---")
st.subheader("📊 Oyun Özeti")

if len(st.session_state.history) == 0:
    st.write("Henüz tur oynanmadı.")
else:
    df_hist = pd.DataFrame(st.session_state.history)

    # Sade özet tablo
    show_cols = [
        "Tur", "KuyrukOlay", "CDS", "TR_Tahvil_Faiz",
        "Portföy_Getiri", "VaR95", "Portföy_Değeri",
        "Ağırlık_TR", "Ağırlık_US", "Ağırlık_Borsa", "Ağırlık_Nakit"
    ]
    df_show = df_hist[show_cols].copy()

    # Formatlama için kopya
    df_show["TR_Tahvil_Faiz"] = df_show["TR_Tahvil_Faiz"] * 100
    df_show["Portföy_Getiri"] = df_show["Portföy_Getiri"] * 100
    df_show["VaR95"] = df_show["VaR95"] * 100
    for c in ["Ağırlık_TR", "Ağırlık_US", "Ağırlık_Borsa", "Ağırlık_Nakit"]:
        df_show[c] = df_show[c] * 100

    st.dataframe(
        df_show.style.format({
            "TR_Tahvil_Faiz": "{:.1f}%",
            "Portföy_Getiri": "{:.2f}%",
            "VaR95": "{:.2f}%",
            "Portföy_Değeri": "{:,.0f}",
            "Ağırlık_TR": "{:.0f}%",
            "Ağırlık_US": "{:.0f}%",
            "Ağırlık_Borsa": "{:.0f}%",
            "Ağırlık_Nakit": "{:.0f}%",
        }),
        use_container_width=True
    )

    final_value = st.session_state.capital
    st.metric("Final Portföy Değeri", f"{final_value:,.0f} TL")

    # CSV indir
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Sonuçları CSV indir", data=csv, file_name="bireysel_portfoy_oyunu_sonuclar.csv", mime="text/csv")
