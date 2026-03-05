# app.py
# Borsa Uygulamaları – CDS / Faiz / Tahvil / VaR / Kuyruk Riski Sınıf Oyunu
# Streamlit app – 4 grup portföy oyunu

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Borsa Uygulamaları | CDS–Faiz Oyunu", layout="wide")

# -----------------------------
# Yardımcı Fonksiyonlar
# -----------------------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def bond_price_from_yield(y, duration=1.8, face=100.0):
    """
    Basit fiyat yaklaşımı:
    Fiyat ≈ Face / (1 + y)^duration
    y: yıllık faiz oranı (0.40 gibi)
    """
    return face / ((1.0 + max(y, 1e-6)) ** duration)

def compute_tr_yield(policy_rate, cds_bps, inf_exp):
    """
    Türkiye 2Y tahvil faizi (temsili):
    y = politika + enflasyon_beklentisi_katmanı + risk_primi
    risk_primi ≈ cds_bps / 10000 * k
    """
    k = 1.10  # CDS'in tahvile yansıma katsayısı (temsili)
    risk_premium = (cds_bps / 10000.0) * k
    inflation_layer = 0.30 * inf_exp  # enflasyon beklentisinin bir kısmı faize biner (temsili)
    y = policy_rate + inflation_layer + risk_premium
    return max(y, 0.0)

def tail_prob_from_cds(cds_bps):
    """
    CDS arttıkça kuyruk olay olasılığı artsın (temsili).
    100 bps -> ~1% ; 1000 bps -> ~10% aralığı gibi.
    """
    p = 0.005 + (cds_bps / 10000.0) * 0.10
    return clamp(p, 0.01, 0.15)

def simple_var_95(port_returns):
    """
    Tarihsel VaR(%95): en kötü %5 dilimin eşiği
    Return dizisi: yüzde değil, oran (0.02 gibi)
    VaR pozitif kayıp olarak raporlanır.
    """
    q = np.quantile(port_returns, 0.05)
    return max(0.0, -q)

def simulate_equity_return(rng, base_mu, base_sigma, tail_event):
    """
    Hisse getirisi:
    normal gün: N(mu, sigma)
    kuyruk gün: ekstra negatif şok
    """
    r = rng.normal(base_mu, base_sigma)
    if tail_event:
        r += rng.normal(-0.18, 0.06)  # kuyruk şoku (temsili)
    return r

def simulate_fx_return(rng, cds_bps, tail_event):
    """
    Kur getirisi (USD/TRY gibi düşünülebilir):
    CDS yükseldikçe yukarı yönlü baskı artsın (temsili).
    """
    drift = 0.002 + (cds_bps / 10000.0) * 0.02  # aylık/period drift (temsili)
    vol = 0.015 + (cds_bps / 10000.0) * 0.05
    r = rng.normal(drift, vol)
    if tail_event:
        r += rng.normal(0.10, 0.05)  # kuyrukta kur sıçraması
    return r

# -----------------------------
# Senaryo Turları (İstersen burada sayıları değiştir)
# -----------------------------
DEFAULT_ROUNDS = [
    {"round": 1, "headline": "Sakin dönem: risk algısı düşük", "policy": 0.35, "cds": 250, "inf_exp": 0.30},
    {"round": 2, "headline": "Enflasyon beklentisi yükseliyor", "policy": 0.35, "cds": 320, "inf_exp": 0.45},
    {"round": 3, "headline": "Risk şoku: CDS hızla yükseldi", "policy": 0.40, "cds": 600, "inf_exp": 0.50},
    {"round": 4, "headline": "Küresel sıkılaşma + belirsizlik", "policy": 0.45, "cds": 750, "inf_exp": 0.55},
    {"round": 5, "headline": "İyileşme: CDS düşüyor, beklentiler toparlanıyor", "policy": 0.40, "cds": 420, "inf_exp": 0.40},
]

ASSETS = ["TR_2Y_BOND", "US_BOND", "EQUITY", "CASH"]

ASSET_LABELS = {
    "TR_2Y_BOND": "Türkiye 2Y Tahvil",
    "US_BOND": "ABD Tahvil (Düşük risk)",
    "EQUITY": "Borsa Endeksi (Riskli)",
    "CASH": "Nakit",
}

# -----------------------------
# Session State
# -----------------------------
if "rounds" not in st.session_state:
    st.session_state.rounds = DEFAULT_ROUNDS

if "current_round_idx" not in st.session_state:
    st.session_state.current_round_idx = 0

if "teams" not in st.session_state:
    st.session_state.teams = ["Grup 1", "Grup 2", "Grup 3", "Grup 4"]

if "initial_capital" not in st.session_state:
    st.session_state.initial_capital = 1_000_000

if "capital" not in st.session_state:
    st.session_state.capital = {t: st.session_state.initial_capital for t in st.session_state.teams}

if "allocations" not in st.session_state:
    # allocations[round][team] = dict(asset->weight)
    st.session_state.allocations = {}

if "history" not in st.session_state:
    # store realized outcomes each round
    st.session_state.history = []

if "seed" not in st.session_state:
    st.session_state.seed = 42

# -----------------------------
# Sidebar – Kontroller
# -----------------------------
st.sidebar.title("🎮 Oyun Kontrol Paneli")

st.sidebar.number_input("Rastgelelik tohumu (seed)", min_value=0, max_value=999999, value=st.session_state.seed, key="seed")

if st.sidebar.button("🔄 Oyunu Sıfırla"):
    st.session_state.current_round_idx = 0
    st.session_state.capital = {t: st.session_state.initial_capital for t in st.session_state.teams}
    st.session_state.allocations = {}
    st.session_state.history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Oyun Mantığı (kısa)")
st.sidebar.write(
    "- CDS ↑ → risk primi ↑ → TR tahvil faizi ↑\n"
    "- Faiz ↑ → tahvil fiyatı ↓\n"
    "- Kuyruk olay olasılığı CDS ile artar\n"
    "- VaR: en kötü %5 eşiği (temsili)"
)

# -----------------------------
# Üst Bilgi
# -----------------------------
st.title("Borsa Uygulamaları | CDS–Faiz–Tahvil–VaR Grup Oyunu")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.write("**Amaç:** CDS, politika faizi ve enflasyon beklentisi değiştikçe yatırımcı gibi karar verip (portföy) sonuçları görmek.")
with colB:
    st.metric("Başlangıç Sermayesi (her grup)", f"{st.session_state.initial_capital:,.0f} TL")
with colC:
    st.metric("Toplam Tur", len(st.session_state.rounds))

# -----------------------------
# Mevcut Tur Bilgisi
# -----------------------------
idx = st.session_state.current_round_idx
if idx >= len(st.session_state.rounds):
    st.success("Oyun bitti. Aşağıdaki tabloda sonuçları görebilirsiniz.")
    current = None
else:
    current = st.session_state.rounds[idx]

if current is not None:
    st.subheader(f"Tur {current['round']}: {current['headline']}")

    policy = current["policy"]
    cds = current["cds"]
    inf_exp = current["inf_exp"]

    tr_yield = compute_tr_yield(policy, cds, inf_exp)
    us_yield = 0.05  # temsili sabit
    tail_p = tail_prob_from_cds(cds)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Politika faizi", f"%{policy*100:.1f}")
    c2.metric("CDS", f"{cds} bps")
    c3.metric("Enflasyon bekl.", f"%{inf_exp*100:.1f}")
    c4.metric("TR 2Y tahvil faizi (temsili)", f"%{tr_yield*100:.1f}")
    c5.metric("Kuyruk olay olasılığı", f"%{tail_p*100:.1f}")

    st.caption("Not: Hesaplamalar öğretim amaçlı temsili bir modeldir.")

# -----------------------------
# Portföy Girişi (4 grup)
# -----------------------------
st.markdown("## 1) Gruplar Portföy Dağılımını Girer")

info = st.info(
    "Her grubun toplamı **%100** olmalı. "
    "TR tahvil yüksek faiz sunabilir ama CDS artınca **tahvil fiyatı düşebilir** ve kuyruk riskinde şok yaşanabilir."
)

alloc_table = []
for team in st.session_state.teams:
    st.markdown(f"### {team}")
    w1 = st.slider(f"{team} – Türkiye 2Y Tahvil (%)", 0, 100, 40, key=f"{idx}_{team}_tr")
    w2 = st.slider(f"{team} – ABD Tahvil (%)", 0, 100, 20, key=f"{idx}_{team}_us")
    w3 = st.slider(f"{team} – Borsa Endeksi (%)", 0, 100, 30, key=f"{idx}_{team}_eq")
    w4 = st.slider(f"{team} – Nakit (%)", 0, 100, 10, key=f"{idx}_{team}_cash")

    total = w1 + w2 + w3 + w4
    st.write(f"Toplam: **%{total}**")
    if total != 100:
        st.warning("Toplam %100 olmalı (slayt mantığı: portföy bütçesi).")
    alloc_table.append([team, w1, w2, w3, w4, total])

df_alloc = pd.DataFrame(alloc_table, columns=["Grup", "TR Tahvil", "US Tahvil", "Borsa", "Nakit", "Toplam"])

# -----------------------------
# Tur Sonuçlarını Üret
# -----------------------------
st.markdown("## 2) Tur Sonucunu Çalıştır")

if current is None:
    st.stop()

can_run = df_alloc["Toplam"].eq(100).all()
if not can_run:
    st.error("Tüm grupların toplamı %100 olmadan tur çalıştırılamaz.")
else:
    if st.button("▶️ Bu Turu Çalıştır (sonucu üret)"):
        rng = np.random.default_rng(st.session_state.seed + current["round"] * 101)

        # Kuyruk olayı gerçekleşsin mi?
        tail_event = rng.random() < tail_prob_from_cds(current["cds"])

        # Bu tur için varlık getirileri (temsili)
        # TR tahvil getirisi: kupon benzeri + fiyat etkisi (faiz değişimi ile)
        # Önce "önceki tur" faizini alıp fiyat değişimine bakacağız
        if idx == 0:
            prev_tr_yield = tr_yield  # ilk turda "önceki" yok
        else:
            prev = st.session_state.rounds[idx - 1]
            prev_tr_yield = compute_tr_yield(prev["policy"], prev["cds"], prev["inf_exp"])

        prev_price = bond_price_from_yield(prev_tr_yield)
        curr_price = bond_price_from_yield(tr_yield)
        price_return = (curr_price - prev_price) / prev_price

        # Basit "kupon" getirisi: tur başına (temsili) tr_yield/4
        coupon_return = tr_yield / 4.0

        tr_bond_return = coupon_return + price_return
        if tail_event:
            tr_bond_return += rng.normal(-0.06, 0.03)  # kuyrukta tahvil de zarar görebilir

        us_bond_return = (us_yield / 4.0) + rng.normal(0.0, 0.003)
        equity_return = simulate_equity_return(rng, base_mu=0.01, base_sigma=0.05, tail_event=tail_event)
        cash_return = 0.0

        # Her grubun portföy getirisi ve yeni sermaye
        round_rows = []
        for _, row in df_alloc.iterrows():
            team = row["Grup"]
            w = np.array([row["TR Tahvil"], row["US Tahvil"], row["Borsa"], row["Nakit"]], dtype=float) / 100.0

            asset_returns = np.array([tr_bond_return, us_bond_return, equity_return, cash_return])
            port_return = float((w * asset_returns).sum())

            # Basit VaR tahmini için aynı turda 200 senaryo simüle edelim
            sims = []
            for _s in range(200):
                te = (rng.random() < tail_prob_from_cds(current["cds"]))
                tr_sim = (tr_yield / 4.0) + rng.normal(price_return, 0.01) + (rng.normal(-0.06, 0.03) if te else 0.0)
                us_sim = (us_yield / 4.0) + rng.normal(0.0, 0.003)
                eq_sim = simulate_equity_return(rng, 0.01, 0.05, te)
                ca_sim = 0.0
                sims.append(float((w * np.array([tr_sim, us_sim, eq_sim, ca_sim])).sum()))
            var95 = simple_var_95(np.array(sims))

            old_cap = st.session_state.capital[team]
            new_cap = old_cap * (1.0 + port_return)
            st.session_state.capital[team] = new_cap

            round_rows.append({
                "Tur": current["round"],
                "Grup": team,
                "Portföy Getirisi": port_return,
                "VaR(95%) (temsili)": var95,
                "Sermaye (TL)": new_cap
            })

        # Kaydet
        st.session_state.history.append({
            "round": current["round"],
            "tail_event": tail_event,
            "policy": policy,
            "cds": cds,
            "inf_exp": inf_exp,
            "tr_yield": tr_yield,
            "tr_bond_return": tr_bond_return,
            "us_bond_return": us_bond_return,
            "equity_return": equity_return,
        })

        # Sonuç göster
        st.success("Tur çalıştırıldı. Sonuçlar aşağıda.")
        out = pd.DataFrame(round_rows)

        st.markdown("### Tur Varlık Getirileri (temsili)")
        st.write(pd.DataFrame([{
            "Kuyruk Olayı": "EVET" if tail_event else "HAYIR",
            "TR Tahvil Getirisi": tr_bond_return,
            "US Tahvil Getirisi": us_bond_return,
            "Borsa Getirisi": equity_return,
            "Nakit Getirisi": 0.0
        }]))

        st.markdown("### Grup Sonuçları")
        st.dataframe(
            out.style.format({
                "Portföy Getirisi": "{:.2%}",
                "VaR(95%) (temsili)": "{:.2%}",
                "Sermaye (TL)": "{:,.0f}"
            }),
            use_container_width=True
        )

        # Sonraki tura geç
        st.session_state.current_round_idx += 1

        st.info("Tartışma önerisi: CDS ↑ olduysa tahvil faizi neden yükseldi, fakat tahvil fiyatı neden düşmüş olabilir? VaR neyi ölçüyor, neyi ölçmüyor?")
        st.rerun()

# -----------------------------
# Skor Tablosu
# -----------------------------
st.markdown("## Skor Tablosu (Güncel Sermaye)")
score_df = pd.DataFrame([
    {"Grup": t, "Sermaye (TL)": st.session_state.capital[t]}
    for t in st.session_state.teams
]).sort_values("Sermaye (TL)", ascending=False)

st.dataframe(score_df.style.format({"Sermaye (TL)": "{:,.0f}"}), use_container_width=True)

# -----------------------------
# Oyun Geçmişi
# -----------------------------
with st.expander("📜 Oyun Geçmişi (tur bazlı)"):
    if len(st.session_state.history) == 0:
        st.write("Henüz tur çalıştırılmadı.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Geçmişi CSV indir", data=csv, file_name="oyun_gecmisi.csv", mime="text/csv")
