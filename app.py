import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Risk mi Getiri mi? | Portföy Oyunu", layout="wide")

STARTING_CAPITAL = 100000
N_ROUNDS = 5

ASSETS = ["TR","US","EQ","FX","CASH"]

ASSET_NAMES = {
"TR":"Türkiye 2Y Tahvil",
"US":"ABD Tahvil",
"EQ":"Borsa Endeksi",
"FX":"USD/TRY",
"CASH":"Nakit"
}

BASE = {
"TR":{"mu":0.02,"sigma":0.02},
"US":{"mu":0.01,"sigma":0.01},
"EQ":{"mu":0.03,"sigma":0.06},
"FX":{"mu":0.015,"sigma":0.04},
"CASH":{"mu":0.0,"sigma":0.0}
}

PIYASA_KOSULLARI = {

"Sakin":{
"TR":{"mu_add":0.0,"sigma_mult":1},
"US":{"mu_add":0.0,"sigma_mult":1},
"EQ":{"mu_add":0.0,"sigma_mult":1},
"FX":{"mu_add":0.0,"sigma_mult":1}
},

"Enflasyon Baskısı":{
"TR":{"mu_add":-0.002,"sigma_mult":1.2},
"US":{"mu_add":0,"sigma_mult":1},
"EQ":{"mu_add":-0.004,"sigma_mult":1.15},
"FX":{"mu_add":0.004,"sigma_mult":1.15}
},

"Risk Şoku":{
"TR":{"mu_add":-0.01,"sigma_mult":1.6},
"US":{"mu_add":0.001,"sigma_mult":1.1},
"EQ":{"mu_add":-0.015,"sigma_mult":1.45},
"FX":{"mu_add":0.012,"sigma_mult":1.6}
},

"Stres":{
"TR":{"mu_add":-0.006,"sigma_mult":1.35},
"US":{"mu_add":0.001,"sigma_mult":1.1},
"EQ":{"mu_add":-0.01,"sigma_mult":1.3},
"FX":{"mu_add":0.008,"sigma_mult":1.35}
},

"İyileşme":{
"TR":{"mu_add":0.003,"sigma_mult":0.95},
"US":{"mu_add":0,"sigma_mult":1},
"EQ":{"mu_add":0.006,"sigma_mult":0.9},
"FX":{"mu_add":-0.004,"sigma_mult":0.95}
}

}

ROUNDS = [

{"tur":1,"piyasa":"Sakin","cds":250},
{"tur":2,"piyasa":"Enflasyon Baskısı","cds":350},
{"tur":3,"piyasa":"Risk Şoku","cds":650},
{"tur":4,"piyasa":"Stres","cds":800},
{"tur":5,"piyasa":"İyileşme","cds":420}

]

def dynamic_params(piyasa,cds):

    out={a:BASE[a].copy() for a in ASSETS}
    r=PIYASA_KOSULLARI[piyasa]

    for a in ["TR","US","EQ","FX"]:
        out[a]["mu"]+=r[a]["mu_add"]
        out[a]["sigma"]*=r[a]["sigma_mult"]

    cds_scale=cds/10000

    out["EQ"]["mu"]-=cds_scale*0.08
    out["EQ"]["sigma"]*=1+cds_scale

    out["TR"]["mu"]-=cds_scale*0.02
    out["FX"]["mu"]+=cds_scale*0.1

    return out

def simulate_returns(dyn):

    r={}

    for a in ASSETS:

        if a=="CASH":
            r[a]=0
        else:
            r[a]=np.random.normal(dyn[a]["mu"],dyn[a]["sigma"])

    return r

def portfolio_expected(weights,dyn):

    mu=0
    var=0

    for a in ASSETS:
        mu+=weights[a]*dyn[a]["mu"]
        var+=(weights[a]**2)*(dyn[a]["sigma"]**2)

    return mu,np.sqrt(var)

def pk_card_html(label,value):

    colors={
    "Sakin":"#2ecc71",
    "Enflasyon Baskısı":"#f1c40f",
    "Risk Şoku":"#e67e22",
    "Stres":"#e74c3c",
    "İyileşme":"#3498db"
    }

    color=colors.get(value,"gray")

    return f"""
<div style="border-left:8px solid {color};
padding:12px;
border-radius:6px;
background:#f8f9fa">

<b>{label}</b><br>
<span style="font-size:1.50rem">{value}</span>

</div>
"""

def risk_label_and_bar(eq_fx):

    score=int(eq_fx*100)

    if score<30:
        label="Düşük"
    elif score<60:
        label="Orta"
    else:
        label="Yüksek"

    filled=int(score/10)

    bar="█"*filled+"░"*(10-filled)

    return label,bar

if "capital" not in st.session_state:
    st.session_state.capital=STARTING_CAPITAL

if "tur" not in st.session_state:
    st.session_state.tur=0

if "history" not in st.session_state:
    st.session_state.history=[]

st.title("Risk mi Getiri mi? – Portföy Oyunu")

st.metric("Portföy Değeri",f"{st.session_state.capital:,.0f} TL")

if st.session_state.tur>=N_ROUNDS:

    st.success("Oyun tamamlandı")

    df=pd.DataFrame(st.session_state.history)

    st.dataframe(df)

    st.line_chart(df[["Portföy"]])

    st.stop()

r=ROUNDS[st.session_state.tur]

st.subheader("Piyasa Kartı")

st.markdown(pk_card_html("Piyasa Koşulları",r["piyasa"]),unsafe_allow_html=True)

dyn=dynamic_params(r["piyasa"],r["cds"])

st.write("Beklenen Getiriler")

df_dyn=pd.DataFrame([
{"Varlık":ASSET_NAMES[a],"Beklenen":dyn[a]["mu"]}
for a in ASSETS
])

st.dataframe(df_dyn)

st.subheader("Portföy Dağılımı")

pct_tr=st.number_input("TR Tahvil %",0,100,30)
pct_us=st.number_input("US Tahvil %",0,100,20)
pct_eq=st.number_input("Borsa %",0,100,30)
pct_fx=st.number_input("Kur %",0,100,10)
pct_cash=st.number_input("Nakit %",0,100,10)

total=pct_tr+pct_us+pct_eq+pct_fx+pct_cash

st.write("Toplam",total)

if total!=100:
    st.error("Toplam %100 olmalı")

weights={
"TR":pct_tr/100,
"US":pct_us/100,
"EQ":pct_eq/100,
"FX":pct_fx/100,
"CASH":pct_cash/100
}

exp_mu,exp_sigma=portfolio_expected(weights,dyn)

st.metric("Beklenen Getiri",f"{exp_mu*100:.2f}%")

eq_fx=weights["EQ"]+weights["FX"]

lab,bar=risk_label_and_bar(eq_fx)

score=int(eq_fx*100)

st.markdown(f"""
Portföy Risk Göstergesi

{bar}

Risk Seviyesi: **{lab}**

Borsa + Kur Ağırlığı: **%{score}**
""")

if st.button("Turu Oyna"):

    rets=simulate_returns(dyn)

    port=sum(weights[a]*rets[a] for a in ASSETS)

    new_val=st.session_state.capital*(1+port)

    contrib={a:weights[a]*rets[a] for a in ASSETS}

    best_asset=max(contrib,key=contrib.get)
    worst_asset=min(contrib,key=contrib.get)

    st.write("En çok katkı:",ASSET_NAMES[best_asset],round(contrib[best_asset]*100,2),"%")

    st.write("En olumsuz katkı:",ASSET_NAMES[worst_asset],round(contrib[worst_asset]*100,2),"%")

    st.session_state.capital=new_val

    st.session_state.history.append({

    "Tur":r["tur"],
    "Piyasa":r["piyasa"],
    "Portföy":new_val

    })

    st.session_state.tur+=1

    st.rerun()
