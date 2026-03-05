import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Risk mi Getiri mi? | Portföy Oyunu", layout="wide")

# -----------------------------
# FONT KÜÇÜLTME (Piyasa Kartı)
# -----------------------------
st.markdown("""
<style>
.piyasa_karti {
  font-size: 0.85rem;
  line-height: 1.25;
}

.piyasa_karti p,
.piyasa_karti li,
.piyasa_karti .stMarkdown {
  font-size: 0.85rem !important;
}

.piyasa_karti [data-testid="stMetricLabel"] {
  font-size: 0.75rem !important;
}

.piyasa_karti [data-testid="stMetricValue"] {
  font-size: 1.05rem !important;
}

.piyasa_karti [data-testid="stDataFrame"] {
  font-size: 0.80rem !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Oyun Parametreleri
# -----------------------------

STARTING_CAPITAL = 100000
N_ROUNDS = 5

ASSETS = ["TR", "US", "EQ", "FX", "CASH"]

ASSET_NAMES = {
    "TR": "Türkiye Tahvil",
    "US": "ABD Tahvil",
    "EQ": "Borsa",
    "FX": "USD/TRY",
    "CASH": "Nakit"
}

BASE = {
    "TR": {"mu":0.02,"sigma":0.02},
    "US": {"mu":0.01,"sigma":0.01},
    "EQ": {"mu":0.03,"sigma":0.05},
    "FX": {"mu":0.02,"sigma":0.04},
    "CASH":{"mu":0,"sigma":0}
}


ROUNDS = [

{"tur":1,"rejim":"Sakin",
"haber":"Piyasalarda sakin dönem",
"policy":0.35,"cds":250,"inf":0.30},

{"tur":2,"rejim":"Enflasyon",
"haber":"Enflasyon beklentisi yükseldi",
"policy":0.35,"cds":350,"inf":0.45},

{"tur":3,"rejim":"Risk Şoku",
"haber":"Risk algısı bozuldu, CDS yükseldi",
"policy":0.40,"cds":650,"inf":0.50},

{"tur":4,"rejim":"Stres",
"haber":"Finansal belirsizlik artıyor",
"policy":0.45,"cds":800,"inf":0.55},

{"tur":5,"rejim":"İyileşme",
"haber":"Riskler azalmaya başladı",
"policy":0.40,"cds":420,"inf":0.40},

]


# -----------------------------
# Fonksiyonlar
# -----------------------------

def tr_yield(policy, cds, inf):

    risk = cds/10000
    return policy + 0.3*inf + risk


def get_params(cds):

    params = BASE.copy()

    cds_scale = cds/10000

    params["EQ"]["mu"] -= cds_scale*0.08
    params["EQ"]["sigma"] *= (1+cds_scale)

    params["FX"]["mu"] += cds_scale*0.1
    params["FX"]["sigma"] *= (1+cds_scale)

    return params



def simulate_returns(params):

    r={}

    for a in ASSETS:

        mu=params[a]["mu"]
        sig=params[a]["sigma"]

        r[a]=np.random.normal(mu,sig)

    return r



# -----------------------------
# Session State
# -----------------------------

if "capital" not in st.session_state:
    st.session_state.capital = STARTING_CAPITAL

if "tur" not in st.session_state:
    st.session_state.tur = 0

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Başlık
# -----------------------------

st.title("🎮 Risk mi Getiri mi?")

col1,col2 = st.columns(2)

col1.metric("Portföy Değeri",f"{st.session_state.capital:,.0f} TL")

if col2.button("Oyunu Sıfırla"):

    st.session_state.capital = STARTING_CAPITAL
    st.session_state.tur = 0
    st.session_state.history = []
    st.rerun()


st.divider()


left,right = st.columns([1.2,0.8])


# -----------------------------
# SOL TARAF
# -----------------------------

with left:

    st.subheader("1) Varlık Kartları")

    df=pd.DataFrame(BASE).T
    st.dataframe(df)


    st.subheader("2) Bu Tur Piyasa Kartı")

    st.markdown('<div class="piyasa_karti">',unsafe_allow_html=True)

    if st.session_state.tur < N_ROUNDS:

        r = ROUNDS[st.session_state.tur]

        y = tr_yield(r["policy"],r["cds"],r["inf"])

        c1,c2,c3,c4,c5=st.columns(5)

        c1.metric("Tur",f"{r['tur']}/{N_ROUNDS}")
        c2.metric("Rejim",r["rejim"])
        c3.metric("Politika",f"%{r['policy']*100:.1f}")
        c4.metric("CDS",f"{r['cds']} bps")
        c5.metric("Enflasyon",f"%{r['inf']*100:.1f}")

        st.write("Haber:",r["haber"])

        st.metric("TR Tahvil Faizi",f"%{y*100:.1f}")

    else:

        st.success("Oyun tamamlandı")

    st.markdown("</div>",unsafe_allow_html=True)


# -----------------------------
# SAĞ TARAF
# -----------------------------

with right:

    st.subheader("3) Karar Ver ve Oyna")

    tr = st.number_input("TR Tahvil %",0,100,30)
    us = st.number_input("US Tahvil %",0,100,20)
    eq = st.number_input("Borsa %",0,100,30)
    fx = st.number_input("USD/TRY %",0,100,10)
    cash = st.number_input("Nakit %",0,100,10)

    total = tr+us+eq+fx+cash

    st.write("Toplam:",total,"%")

    if st.button("Turu Oyna") and total==100 and st.session_state.tur < N_ROUNDS:

        r = ROUNDS[st.session_state.tur]

        params = get_params(r["cds"])

        ret = simulate_returns(params)

        weights={
        "TR":tr/100,
        "US":us/100,
        "EQ":eq/100,
        "FX":fx/100,
        "CASH":cash/100
        }

        port_r=0

        for a in ASSETS:

            port_r+=weights[a]*ret[a]

        st.session_state.capital *= (1+port_r)

        st.session_state.history.append({

        "tur":r["tur"],
        "getiri":port_r,
        "portfoy":st.session_state.capital

        })

        st.session_state.tur+=1

        st.rerun()


# -----------------------------
# SONUÇ
# -----------------------------

st.divider()

st.subheader("Sonuçlar")

if len(st.session_state.history)>0:

    df=pd.DataFrame(st.session_state.history)

    st.dataframe(df)
