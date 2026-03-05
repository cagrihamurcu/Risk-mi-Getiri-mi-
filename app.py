# app.py
# Borsa Uygulamaları – Bireysel Portföy Oyunu (Gerçek veri YOK)
#
# EKLENENLER (gerçek veri çekmeden):
# 1) Beklenen Getiri vs Gerçekleşen Getiri (tur sonunda tablo)
# 2) Sürpriz Piyasa Şokları (olasılıklı; kontrollü)
# 3) Portföy Risk Göstergesi (EQ% + FX% bar + etiket)
# 4) Karşılaştırma Portföyleri (benchmark): %100 TR Tahvil, %60 Borsa-40 TR Tahvil, %100 Nakit
# 5) Tur sonu otomatik yorum
# 6) HER SENARYO ÖNCESİ: Piyasa yorumu + Referans portföy önerisi (Hızlı soru YOK)
#
# Leaderboard YOK.
# matplotlib YOK (Streamlit Cloud uyumlu).
#
# Çalıştır:
#   pip install streamlit numpy pandas
#   streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Risk mi Getiri mi? | Portföy Oyunu", layout="wide")

# -------------------------------------------------
# CSS: Piyasa kartı altı küçük font + Piyasa Koşulları değeri küçük kart
# -------------------------------------------------
st.markdown(
    """
<style>
/* 2) Bu Tur Piyasa Kartı altı */
.piyasa_karti { font-size: 0.84rem; line-height: 1.25; }
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
.pk_label { font-size: 0.70rem; opacity: 0.75; margin-bottom: 6px; }
.pk_value {
  font-size: 0.88rem;
  font-weight: 700;
  line-height: 1.1;
  word-break: break-word;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# OYUN PARAMETRELERİ
# -------------------------------------------------
STARTING_CAPITAL = 100_000
N_ROUNDS = 5
SHOCK_PROB = 0.20  # her tur sürpriz şok olasılığı

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

# Senaryo öncesi bilgilendirme + referans portföy (tur başlamadan)
SCENARIO_GUIDE = {
    "Sakin": {
        "comment": (
            "Piyasa oynaklığı düşük ve CDS seviyesi sınırlı. Risk algısı dengeli olduğu için "
            "hem tahvil hem hisse tarafında pozisyon alınabilir. Kur tarafında büyük sıçrama beklentisi zayıftır."
        ),
        "portfolio": {"TR": 30, "US": 20, "EQ": 35, "FX": 10, "CASH": 5},
    },
    "Enflasyon Baskısı": {
        "comment": (
            "Enflasyon beklentisi yükseliyor. Bu ortamda tahvil piyasasında baskı artabilir; "
            "borsa dalgalanabilir ve kur yukarı tepki verebilir. Daha dengeli ve korumacı dağılım tercih edilebilir."
        ),
        "portfolio": {"TR": 25, "US": 25, "EQ": 25, "FX": 15, "CASH": 10},
    },
    "Risk Şoku": {
        "comment": (
            "CDS hızlı yükselmiştir; ülke risk algısı bozulmuştur. Riskten kaçış dönemlerinde "
            "borsada satış baskısı ve kurda oynaklık artışı görülebilir. Koruma amaçlı tahvil ve nakit payı artırılabilir."
        ),
        "portfolio": {"TR": 30, "US": 30, "EQ": 15, "FX": 15, "CASH": 10},
    },
    "Stres": {
        "comment": (
            "Belirsizlik yüksektir ve risk primi baskındır. Bu tip dönemlerde yatırımcılar "
            "genellikle daha güvenli varlıklara yönelir; riskli varlıklarda ani hareketler görülebilir. Korumacı duruş öne çıkar."
        ),
        "portfolio": {"TR": 35, "US": 30, "EQ": 10, "FX": 15, "CASH": 10},
    },
    "İyileşme": {
        "comment": (
            "Risk primi gerilemeye başlamıştır. Risk iştahı kademeli artabilir ve borsa toparlanma eğilimi gösterebilir. "
            "Daha riskli varlıklara (özellikle hisse) ağırlık artırma eğilimi görülebilir."
        ),
        "portfolio": {"TR": 25, "US": 15, "EQ": 45, "FX": 10, "CASH": 5},
    },
}

SHOCKS = [
    {
        "name": "Politika Sıkılaşması Şoku",
        "desc": "Merkez bankasından beklenenden sert sıkılaşma sinyali.",
        "impacts": {"TR": -0.030, "EQ": -0.050, "FX": +0.020, "US": +0.003, "CASH": 0.0},
    },
    {
        "name": "Küresel Riskten Kaçış",
        "desc": "Global risk iştahı düştü, güvenli liman talebi arttı.",
        "impacts": {"TR": -0.020, "EQ": -0.070, "FX": +0.040, "US": +0.008, "CASH": 0.0},
    },
    {
        "name": "Risk İştahı Artışı",
        "desc": "Beklentiler iyileşti, riskli varlıklara giriş hızlandı.",
        "impacts": {"TR": +0.010, "EQ": +0.060, "FX": -0.020, "US": -0.002, "CASH": 0.0},
    },
    {
        "name": "Kur Baskısı Şoku",
        "desc": "Kur tarafında ani yukarı hareket beklentisi güçlendi.",
        "impacts": {"TR": -0.015, "EQ": -0.030, "FX": +0.060, "US": +0.002, "CASH": 0.0},
    },
]

BENCHMARKS = {
    "%100 TR Tahvil": {"TR": 1.0, "US": 0.0, "EQ": 0.0, "FX": 0.0, "CASH": 0.0},
    "%60 Borsa / %40 TR Tahvil": {"TR": 0.40, "US": 0.0, "EQ": 0.60, "FX": 0.0, "CASH": 0.0},
    "%100 Nakit": {"TR": 0.0, "US": 0.0, "EQ": 0.0, "FX": 0.0, "CASH": 1.0},
}

# -------------------------------------------------
# FONKSİYONLAR
# -------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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


def pk_card_html(label: str, value: str) -> str:
    return f"""
<div class="pk_card">
  <div class="pk_label">{label}</div>
  <div class="pk_value">{value}</div>
</div>
"""


def risk_label_and_bar(eq_fx_weight: float) -> tuple[str, str]:
    score = int(round(eq_fx_weight * 100))
    if score <= 30:
        lab = "Düşük"
    elif score <= 60:
        lab = "Orta"
    else:
        lab = "Yüksek"
    filled = min(10, max(0, int(round(score / 10))))
    bar = "█" * filled + "░" * (10 - filled)
    return lab, bar


def tr_yield(policy: float, cds_bps: int, inf: float) -> float:
    k = 1.10
    risk_premium = (cds_bps / 10000.0) * k
    inflation_layer = 0.30 * inf
    return max(policy + inflation_layer + risk_premium, 0.0)


def bond_price_from_yield(y: float, duration: float = 1.8, face: float = 100.0) -> float:
    y = max(y, 1e-6)
    return face / ((1.0 + y) ** duration)


def pick_shock(rng: np.random.Generator) -> dict | None:
    if float(rng.random()) > SHOCK_PROB:
        return None
    idx = int(rng.integers(0, len(SHOCKS)))
    return SHOCKS[idx]


def apply_shock(rets: dict, shock: dict | None) -> dict:
    if shock is None:
        return rets
    out = dict(rets)
    for a, add in shock["impacts"].items():
        out[a] = float(out.get(a, 0.0) + add)
    return out


def benchmark_update(prev_vals: dict, realized_rets: dict) -> dict:
    new_vals = {}
    for name, w in BENCHMARKS.items():
        pr = 0.0
        for a in ASSETS:
            pr += w[a] * realized_rets[a]
        new_vals[name] = float(prev_vals[name] * (1.0 + pr))
    return new_vals


def expected_vs_realized_table(dyn: dict, realized_rets: dict) -> pd.DataFrame:
    rows = []
    for a in ASSETS:
        rows.append(
            {
                "Varlık": ASSET_NAMES[a],
                "Beklenen (μ)": f"{dyn[a]['mu']*100:.2f}%",
                "Gerçekleşen": f"{realized_rets[a]*100:.2f}%",
            }
        )
    return pd.DataFrame(rows)


def tur_sonu_yorum(piyasa_kosullari: str, cds_bps: int, weights: dict, realized_rets: dict, tr_price_effect: float, shock: dict | None) -> str:
    contrib = {a: weights[a] * realized_rets[a] for a in ASSETS}
    biggest = max(contrib, key=lambda k: abs(contrib[k]))
    biggest_name = ASSET_NAMES[biggest]
    biggest_points = contrib[biggest] * 100

    eq_fx = weights["EQ"] + weights["FX"]
    risk_lab, _ = risk_label_and_bar(eq_fx)

    lines = []
    lines.append(f"**Piyasa Koşulları:** {piyasa_kosullari} | **CDS:** {cds_bps} bps | **Risk seviyesi:** {risk_lab}")

    if shock is not None:
        lines.append(f"**Sürpriz Şok:** {shock['name']} – {shock['desc']}")

    if cds_bps >= 650:
        lines.append("- CDS yüksek: risk algısı zayıflar → borsada dalgalanma artar, kurda yukarı baskı olasılığı yükselir.")
    elif cds_bps >= 400:
        lines.append("- CDS orta-yüksek: temkinli risk algısı → riskli varlıklarda oynaklık artabilir.")
    else:
        lines.append("- CDS düşük/orta: risk algısı daha sakin → dalgalanma nispeten sınırlı kalabilir.")

    if abs(tr_price_effect) > 0.002:
        direction = "azalttı" if tr_price_effect < 0 else "artırdı"
        lines.append(f"- Faiz değişimi TR tahvil fiyatını **{direction}** (fiyat etkisi: **{tr_price_effect*100:.2f}%**).")

    lines.append(f"- Portföy sonucunu en çok etkileyen kalem: **{biggest_name}** (katkı: **{biggest_points:+.2f} puan**).")

    if eq_fx >= 0.60:
        lines.append("- Borsa+Kur ağırlığı yüksek: şoklarda sert düşüş / iyi haberde sert yükseliş görülebilir.")
    elif weights["TR"] + weights["US"] + weights["CASH"] >= 0.70:
        lines.append("- Tahvil+Nakit ağırlığı yüksek: dalgalanma genelde daha sınırlı olur.")
    else:
        lines.append("- Dengeli dağılım: şoklarda darbeyi azaltıp iyileşmede fırsat yakalama şansını artırır.")

    return "\n".join(lines)


def guide_box(piyasa_kosullari: str):
    g = SCENARIO_GUIDE.get(piyasa_kosullari)
    if not g:
        return
    st.markdown("### 🧭 Tur Öncesi Bilgilendirme")
    st.info(g["comment"])
    st.markdown("### 💡 Referans Portföy (Örnek)")
    port = g["portfolio"]
    guide_df = pd.DataFrame(
        [{"Varlık": ASSET_NAMES[a], "Önerilen %": port[a]} for a in ASSETS]
    )
    st.dataframe(guide_df, use_container_width=True, hide_index=True)


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

if "bench_vals" not in st.session_state:
    st.session_state.bench_vals = {k: float(STARTING_CAPITAL) for k in BENCHMARKS.keys()}
if "bench_hist" not in st.session_state:
    st.session_state.bench_hist = []

for k, v in [("pct_tr", 35), ("pct_us", 20), ("pct_eq", 30), ("pct_fx", 10), ("pct_cash", 5)]:
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------------------------
# ÜST
# -------------------------------------------------
st.title("🎮 Risk mi Getiri mi? – Bireysel Portföy Oyunu")
st.caption("Gerçek veri olmadan, kontrollü senaryolarla: CDS–faiz–tahvil–borsa–kur dinamiklerini deneyimle.")

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
        st.session_state.bench_vals = {k: float(STARTING_CAPITAL) for k in BENCHMARKS.keys()}
        st.session_state.bench_hist = []
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
    st.caption("Normal koşullarda her varlığın **temsili** beklenen getiri (μ) ve oynaklık (σ) varsayımları.")
    df_cards = pd.DataFrame(
        [{
            "Varlık": ASSET_NAMES[a],
            "Baz beklenen getiri (tur)": f"{BASE[a]['mu']*100:.1f}%",
            "Baz oynaklık (tur)": f"{BASE[a]['sigma']*100:.1f}%"
        } for a in ASSETS]
    )
    st.dataframe(df_cards, use_container_width=True, hide_index=True)

    st.subheader("2) Bu Tur Piyasa Kartı")
    st.caption("Bu turda **Piyasa Koşulları + CDS + politika faizi + enflasyon** varlıkların risk/getiri profilini değiştirir.")

    st.markdown('<div class="piyasa_karti">', unsafe_allow_html=True)

    if st.session_state.tur_idx >= N_ROUNDS:
        st.success("Oyun tamamlandı ✅ (Aşağıda sonuçlar var.)")
    else:
        r = ROUNDS[st.session_state.tur_idx]
        tr_y = tr_yield(r["policy"], r["cds"], r["inf"])
        dyn = dynamic_params(r["piyasa_kosullari"], r["cds"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tur", f"{r['tur']}/{N_ROUNDS}")
        c2.markdown(pk_card_html("Piyasa Koşulları", r["piyasa_kosullari"]), unsafe_allow_html=True)
        c3.metric("Politika", f"%{r['policy']*100:.1f}")
        c4.metric("CDS", f"{r['cds']} bps")
        c5.metric("Enflasyon", f"%{r['inf']*100:.1f}")

        st.write(f"**Haber:** {r['haber']}")
        st.metric("TR 2Y tahvil faizi (temsili)", f"%{tr_y*100:.1f}")

        df_dyn = pd.DataFrame([{
            "Varlık": ASSET_NAMES[a],
            "Beklenen getiri (bu tur)": f"{dyn[a]['mu']*100:.2f}%",
            "Oynaklık (bu tur)": f"{dyn[a]['sigma']*100:.2f}%"
        } for a in ASSETS])
        st.dataframe(df_dyn, use_container_width=True, hide_index=True)

        # ✅ Tur öncesi bilgilendirme + referans portföy
        guide_box(r["piyasa_kosullari"])

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# SAĞ
# -------------------------------------------------
with right:
    st.subheader("3) Karar Ver ve Oyna")
    st.caption("Portföy yüzdelerini belirle. Tur sonunda: **beklenen–gerçekleşen**, **şok varsa etkisi**, ve **kısa yorum** oluşur.")

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

        eq_fx = weights["EQ"] + weights["FX"]
        rlab, rbar = risk_label_and_bar(eq_fx)
        st.markdown(f"**Portföy Risk Göstergesi (Borsa% + Kur%)**: `{rbar}`  → **{rlab}** (skor: {int(round(eq_fx*100))})")

        m1, m2 = st.columns(2)
        m1.metric("Beklenen getiri (bu tur)", f"{exp_mu*100:.2f}%")
        m2.metric("Tahmini risk (bu tur)", f"{exp_sigma*100:.2f}%")

        if st.button("▶️ Turu Oyna", disabled=(not ok)):
            rng = np.random.default_rng(st.session_state.seed + r["tur"] * 101)

            # Tahvil fiyat etkisi
            prev_y = st.session_state.prev_tr_yield
            prev_p = bond_price_from_yield(prev_y)
            curr_p = bond_price_from_yield(tr_y)
            tr_price_effect = float((curr_p - prev_p) / prev_p)

            # Baz gerçekleşen getiriler
            rets = simulate_returns(rng, dyn)
            # tahvil: faiz-fiyat etkisini ekle
            rets["TR"] = float(rets["TR"] + tr_price_effect)

            # şok uygula
            shock = pick_shock(rng)
            rets_after = apply_shock(rets, shock)

            # portföy
            port_r = float(sum(weights[a] * rets_after[a] for a in ASSETS))
            new_val = float(st.session_state.capital * (1.0 + port_r))
            st.session_state.capital = new_val

            # benchmark güncelle
            st.session_state.bench_vals = benchmark_update(st.session_state.bench_vals, rets_after)
            st.session_state.bench_hist.append({"Tur": r["tur"], **st.session_state.bench_vals})

            # beklenen vs gerçekleşen
            evr_df = expected_vs_realized_table(dyn, rets_after)

            # yorum
            explanation = tur_sonu_yorum(
                piyasa_kosullari=r["piyasa_kosullari"],
                cds_bps=r["cds"],
                weights=weights,
                realized_rets=rets_after,
                tr_price_effect=tr_price_effect,
                shock=shock
            )

            # kayıt
            st.session_state.history.append({
                "Tur": r["tur"],
                "Piyasa Koşulları": r["piyasa_kosullari"],
                "Haber": r["haber"],
                "Şok": (shock["name"] if shock else ""),
                "CDS": r["cds"],
                "TR_Faiz": tr_y,
                "Tahvil_Fiyat_Etkisi": tr_price_effect,
                "TR_Getiri": rets_after["TR"],
                "US_Getiri": rets_after["US"],
                "Borsa_Getiri": rets_after["EQ"],
                "Kur_Getiri": rets_after["FX"],
                "Portföy_Getiri": port_r,
                "Portföy_Değeri": new_val,
                "A_TR": weights["TR"],
                "A_US": weights["US"],
                "A_EQ": weights["EQ"],
                "A_FX": weights["FX"],
                "A_CASH": weights["CASH"],
                "Bench_%100_TR": st.session_state.bench_vals["%100 TR Tahvil"],
                "Bench_60EQ40TR": st.session_state.bench_vals["%60 Borsa / %40 TR Tahvil"],
                "Bench_%100_Nakit": st.session_state.bench_vals["%100 Nakit"],
                "EVR": evr_df.to_dict(orient="records"),
                "Açıklama": explanation
            })

            st.session_state.prev_tr_yield = tr_y
            st.session_state.tur_idx += 1
            st.rerun()

st.divider()

# -------------------------------------------------
# SONUÇLAR
# -------------------------------------------------
st.subheader("📊 Sonuçlar")

if len(st.session_state.history) == 0:
    st.write("Henüz tur oynanmadı.")
else:
    df = pd.DataFrame(st.session_state.history)

    tab1, tab2, tab3, tab4 = st.tabs(["Özet Tablo", "Detay Tablo", "Açıklamalar", "Karşılaştırmalar"])

    with tab1:
        df_sum = df[["Tur", "Piyasa Koşulları", "CDS", "TR_Faiz", "Portföy_Getiri", "Portföy_Değeri", "Şok"]].copy()
        df_sum["TR_Faiz"] = (df_sum["TR_Faiz"] * 100).round(1).astype(str) + "%"
        df_sum["Portföy_Getiri"] = (df_sum["Portföy_Getiri"] * 100).round(2).astype(str) + "%"
        df_sum["Portföy_Değeri"] = df_sum["Portföy_Değeri"].round(0).astype(int)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

    with tab2:
        cols = [c for c in df.columns if c not in ["EVR", "Açıklama"]]
        df_det = df[cols].copy()
        df_det["TR_Faiz"] = (df_det["TR_Faiz"] * 100).round(1)
        df_det["Tahvil_Fiyat_Etkisi"] = (df_det["Tahvil_Fiyat_Etkisi"] * 100).round(2)
        for c in ["TR_Getiri", "US_Getiri", "Borsa_Getiri", "Kur_Getiri", "Portföy_Getiri"]:
            df_det[c] = (df_det[c] * 100).round(2)
        for c in ["A_TR", "A_US", "A_EQ", "A_FX", "A_CASH"]:
            df_det[c] = (df_det[c] * 100).round(0).astype(int)
        st.dataframe(df_det, use_container_width=True, hide_index=True)

    with tab3:
        for _, row in df.iterrows():
            title = f"Tur {int(row['Tur'])} – {row['Piyasa Koşulları']}"
            if row.get("Şok", ""):
                title += f" | Şok: {row['Şok']}"
            with st.expander(title, expanded=False):
                st.markdown(row["Açıklama"])
                st.markdown("**Beklenen vs Gerçekleşen (bu tur)**")
                st.dataframe(pd.DataFrame(row["EVR"]), use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("📈 Portföy Değeri (Senin)")
        st.line_chart(df[["Tur", "Portföy_Değeri"]].set_index("Tur"))

        if len(st.session_state.bench_hist) > 0:
            bdf = pd.DataFrame(st.session_state.bench_hist).set_index("Tur")
            st.subheader("📉 Benchmark Portföyler (Aynı turlarda)")
            st.line_chart(bdf)

        st.subheader("🏁 Final Karşılaştırma")
        final_rows = [{"Strateji": "Senin Portföyün", "Final Değer": float(st.session_state.capital)}]
        for name in BENCHMARKS.keys():
            final_rows.append({"Strateji": name, "Final Değer": float(st.session_state.bench_vals[name])})
        final_df = pd.DataFrame(final_rows).sort_values("Final Değer", ascending=False)
        final_df["Final Değer"] = final_df["Final Değer"].round(0).astype(int)
        st.dataframe(final_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Sonuçları CSV indir",
        data=df.drop(columns=["EVR"]).to_csv(index=False).encode("utf-8"),
        file_name="borsa_portfoy_oyunu_sonuclar.csv",
        mime="text/csv",
    )
