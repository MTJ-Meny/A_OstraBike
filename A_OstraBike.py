# =====================================================================================
# ZÁVEREČNÝ PROJEKT A: OstraBike (1. 3. 2025 – 31. 10. 2026)
# Otázka: Ako sa systému OstraBike darí po migrácii dát, čo ovplyvňuje dopyt a stav
# bicyklov, ako dopadla zmena ceny odomykania 1. 6. 2026 a kam smerovať investície?
#
# Záver (3 vety):
# 1. Po vyčistení 40 000 riadkov zostalo 39 196 platných jázd (vyradených 2,0 %) a denný
#    dopyt silno súvisí s teplotou (r = 0,85) aj dažďom (v daždivých dňoch >= 1 mm
#    klesá priemer o 43,8 % na 40,3 jázd/deň oproti 71,7 jázd/deň v suchých dňoch),
#    zatiaľ čo sezóna marec–október medziročne vzrástla o +16,1 % (zo 68,5 na 79,6 jázd/deň).
# 2. Zmena ceny odomykania od 1. 6. 2026 znížila v období jún–október jazdy
#    pay_as_you_go o -6,0 % (zo 3 954 na 3 715), ale tržby tejto skupiny vzrástli zo
#    129 243 Kč na 140 920 Kč (ostatné jazdy vzrástli o +21,9 %), a pri bicykloch sa
#    ukázal problém pri konkrétnych kusoch (medián krátkych jázd <= 2 min je 0,0 %,
#    najhorší bicykel B0015 má 99,6 %, odhad ušlých priamych tržieb je 46 322 Kč).
# 3. Prevádzkový tím by mal prioritne opraviť alebo vyradiť problémové bicykle s vysokým
#    podielom krátkych jázd, posilniť ranný presun bicyklov zo staníc s odlivom (napr.
#    Fakultní nemocnice, -251 jázd) do lokalít s prílivom (napr. Nová Ves, +183 jázd)
#    a ďalej rozvíjať 8 nových staníc z apríla 2026, ktoré majú 122 % denného výkonu
#    starých staníc (1,7 vs. 1,4 odchodov/deň).
# =====================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cesta k dátam a priečinok outputs pre grafy a tabuľky uložený priamo v CESTA
CESTA = r"C:\Users\matej\Downloads\A_OstraBike\A_OstraBike"
VYSTUPY = os.path.join(CESTA, "outputs")
os.makedirs(VYSTUPY, exist_ok=True)

# Základné farby a nastavenie grafov z hodín
MODRA = "#2a78d6"
ORANZOVA = "#eb6834"
SEDA = "#52514e"
SVETLE_SEDA = "#c9c8c1"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": SEDA,
    "xtick.color": SEDA,
    "ytick.color": SEDA,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlesize": 11
})

# =====================================================================================
# 1) NAČÍTANIE DÁT A ČISTENIE S LOGOM
# =====================================================================================
jizdy_raw = pd.read_csv(os.path.join(CESTA, "jizdy.csv"))
stanice_raw = pd.read_csv(os.path.join(CESTA, "stanice.csv"))
kola_raw = pd.read_csv(os.path.join(CESTA, "kola.csv"))
uzivatele_raw = pd.read_csv(os.path.join(CESTA, "uzivatele.csv"))
pocasi_raw = pd.read_csv(os.path.join(CESTA, "pocasi.csv"))

print("Riadky raw:", len(jizdy_raw), len(stanice_raw), len(kola_raw), len(uzivatele_raw), len(pocasi_raw))

log_cisteni = []
df = jizdy_raw.copy()
povodny_pocet = len(df)

# Duplicitné riadky: ponecháme prvý výskyt, duplicity by skreslili počty jázd aj tržby
dup_cele = df.duplicated().sum()
df = df.drop_duplicates()
dup_id = df.duplicated(subset=["trip_id"]).sum()
df = df.drop_duplicates(subset=["trip_id"], keep="first")
log_cisteni.append(["Duplicity (celé + trip_id)", int(dup_cele + dup_id), len(df)])
print("Duplicity odstránené:", dup_cele + dup_id, "-> zostáva:", len(df))

# Dva formáty času: nový yyyy-mm-dd hh:mm a starý dd.mm.yyyy hh:mm prevedieme na dátum a čas
start_iso = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M", errors="coerce")
start_stary = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M", errors="coerce")
df["start_dt"] = start_iso.fillna(start_stary)

end_iso = pd.to_datetime(df["end_time"], format="%Y-%m-%d %H:%M", errors="coerce")
end_stary = pd.to_datetime(df["end_time"], format="%d.%m.%Y %H:%M", errors="coerce")
df["end_dt"] = end_iso.fillna(end_stary)

pocet_stary_cas = int(start_stary.notna().sum())
df = df.dropna(subset=["start_dt", "end_dt"]).copy()
log_cisteni.append(["Dva formáty času (zjednotené)", pocet_stary_cas, len(df)])
print("Starý formát času zjednotený:", pocet_stary_cas)

df["dlzka_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60

# Koniec pred začiatkom: záporná dĺžka jazdy je chyba záznamu, vyradíme ju
pred_zaciatkom = int((df["dlzka_min"] < 0).sum())
df = df[df["dlzka_min"] >= 0].copy()
log_cisteni.append(["Koniec pred začiatkom", pred_zaciatkom, len(df)])
print("Vyradené koniec pred začiatkom:", pred_zaciatkom)

# Jazdy dlhšie ako 24 hodín (1440 min): neuzavreté/extrémne výpožičky vyradíme, aby nekazili priemer
nad_24h = int((df["dlzka_min"] > 1440).sum())
df = df[df["dlzka_min"] <= 1440].copy()
log_cisteni.append(["Jazdy dlhšie ako 24 h", nad_24h, len(df)])
print("Vyradené > 24 h:", nad_24h)

# distance_km ako text s čiarkou: čiarku zmeníme na bodku a prevedieme na číslo
ciarky_km = int(df["distance_km"].astype(str).str.contains(",").sum())
df["distance_km"] = pd.to_numeric(df["distance_km"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
df = df[df["distance_km"] >= 0].copy()
log_cisteni.append(["distance_km (čiarka -> číslo)", ciarky_km, len(df)])
print("Opravené distance_km s čiarkou:", ciarky_km)

# Malé písmená v ID: zjednotíme na veľké písmená, aby párovanie tabuliek fungovalo
df["start_station_id"] = df["start_station_id"].astype(str).str.strip().str.upper()
df["end_station_id"] = df["end_station_id"].astype(str).str.strip().str.upper()
df["bike_id"] = df["bike_id"].astype(str).str.strip().str.upper()

# Stanice mimo číselníka (napr. ST999): vyradíme, pretože chýbajú v číselníku staníc
platne_stanice = set(stanice_raw["station_id"].astype(str).str.strip().str.upper())
mimo_stanice = (~df["start_station_id"].isin(platne_stanice)) | (~df["end_station_id"].isin(platne_stanice))
print("Vyradené stanice mimo číselník:", int(mimo_stanice.sum()))
df = df[~mimo_stanice].copy()
log_cisteni.append(["Stanice mimo číselník", int(mimo_stanice.sum()), len(df)])

# Záporné ceny: dobropisy/storná vyradíme, aby nekazili kladnú tržbu
zaporne_ceny = int((df["price_czk"] < 0).sum())
print("Vyradené záporné ceny:", zaporne_ceny)
df = df[df["price_czk"] >= 0].copy()
log_cisteni.append(["Záporné ceny", zaporne_ceny, len(df)])

# Prázdne user_id: overíme, že je len pri turistoch (vlastnosť dát, nie chyba)
print("Prázdne user_id podľa user_type:")
print(df.groupby("user_type")["user_id"].apply(lambda s: s.isna().sum()))

# Počasie a používatelia: chýbajúcu teplotu doplníme interpoláciou, nezmyselné roky narodenia dáme na NaN
pocasi = pocasi_raw.copy()
pocasi["datum"] = pd.to_datetime(pocasi["date"]).dt.date
chyba_teplota = int(pocasi["temp_avg"].isna().sum())
pocasi["temp_avg"] = pocasi["temp_avg"].interpolate(method="linear").bfill().ffill()
print("Doplnených teplôt v počasí:", chyba_teplota)

uzivatele = uzivatele_raw.copy()
uzivatele["user_id"] = uzivatele["user_id"].astype(str).str.strip().str.upper()
rok_nar = np.floor(pd.to_numeric(uzivatele["birth_year"], errors="coerce"))
zle_roky = int(((rok_nar < 1935) | (rok_nar > 2012)).sum())
rok_nar[(rok_nar < 1935) | (rok_nar > 2012)] = np.nan
uzivatele["rok_narodenia"] = rok_nar
print("Vyčistených nezmyselných rokov narodenia:", zle_roky)

log_df = pd.DataFrame(log_cisteni, columns=["krok", "pocet_zmien", "zostava_riadkov"])
log_df.to_csv(os.path.join(VYSTUPY, "00_log_cisteni.csv"), index=False)
print("Čistých jázd:", len(df), "z", povodny_pocet)

# =====================================================================================
# 2) SPOJENIE TABULIEK A KONTROLY ASSERT
# =====================================================================================
stanice = stanice_raw.copy()
stanice["station_id"] = stanice["station_id"].astype(str).str.strip().str.upper()
stanice["je_nova"] = pd.to_datetime(stanice["opened"]) >= pd.Timestamp("2026-04-01")

kola = kola_raw.copy()
kola["bike_id"] = kola["bike_id"].astype(str).str.strip().str.upper()

df["datum"] = df["start_dt"].dt.date
df["rok"] = df["start_dt"].dt.year
df["mesiac"] = df["start_dt"].dt.month
df["hodina"] = df["start_dt"].dt.hour
df["den_v_tyzdni"] = df["start_dt"].dt.day_of_week
df["je_vikend"] = df["den_v_tyzdni"] >= 5

n_pred_merge = len(df)
trzba_pred_merge = df["price_czk"].sum()

df = df.merge(
    stanice[["station_id", "name", "district", "station_type", "lat", "lon", "capacity", "je_nova"]],
    left_on="start_station_id", right_on="station_id", how="left"
).rename(columns={"name": "start_nazov", "district": "start_cast", "station_type": "start_typ", "je_nova": "start_nova"}).drop(columns=["station_id"])
assert len(df) == n_pred_merge

df = df.merge(
    stanice[["station_id", "name", "je_nova"]],
    left_on="end_station_id", right_on="station_id", how="left"
).rename(columns={"name": "ciel_nazov", "je_nova": "ciel_nova"}).drop(columns=["station_id"])
assert len(df) == n_pred_merge

df = df.merge(kola[["bike_id", "bike_type"]], on="bike_id", how="left")
assert len(df) == n_pred_merge

df = df.merge(pocasi[["datum", "temp_avg", "rain_mm", "wind_ms"]], on="datum", how="left")
assert len(df) == n_pred_merge
assert df["temp_avg"].isna().sum() == 0

df["uid_ciste"] = df["user_id"].fillna("").astype(str).str.strip().str.upper()
df = df.merge(uzivatele[["user_id", "rok_narodenia"]], left_on="uid_ciste", right_on="user_id", how="left", suffixes=("", "_u")).drop(columns=["user_id_u", "uid_ciste"])
assert len(df) == n_pred_merge
assert np.isclose(df["price_czk"].sum(), trzba_pred_merge)

# =====================================================================================
# 3) OTÁZKA 1: SEZÓNNY VÝVOJ A POČASIE
# =====================================================================================
denne = df.groupby("datum").agg(jazdy=("trip_id", "count"), trzba_kc=("price_czk", "sum")).reset_index()
kal = pocasi.merge(denne, on="datum", how="left")
kal["jazdy"] = kal["jazdy"].fillna(0)
kal["trzba_kc"] = kal["trzba_kc"].fillna(0)
assert len(kal) == len(pocasi)
assert kal["jazdy"].sum() == len(df)

kal["rok"] = pd.to_datetime(kal["datum"]).dt.year
kal["mesiac"] = pd.to_datetime(kal["datum"]).dt.month
kal["dazdivy"] = kal["rain_mm"] >= 1.0

po_dazdi = kal.groupby("dazdivy")["jazdy"].agg(["count", "mean"])
sucho_priemer = po_dazdi.loc[False, "mean"]
dazd_priemer = po_dazdi.loc[True, "mean"]
sucho_dni = int(po_dazdi.loc[False, "count"])
dazd_dni = int(po_dazdi.loc[True, "count"])
pokles_dazd_pct = (1 - dazd_priemer / sucho_priemer) * 100
kor_teplota = kal["temp_avg"].corr(kal["jazdy"])

medzirocne = kal[kal["mesiac"].between(3, 10)].groupby(["rok", "mesiac"])["jazdy"].mean().unstack(0)
priemer_2025 = kal[(kal["mesiac"].between(3, 10)) & (kal["rok"] == 2025)]["jazdy"].mean()
priemer_2026 = kal[(kal["mesiac"].between(3, 10)) & (kal["rok"] == 2026)]["jazdy"].mean()
zmena_yoy_pct = (priemer_2026 / priemer_2025 - 1) * 100

print(f"Q1 | Sucho (n={sucho_dni}): {sucho_priemer:.1f} | Dážď (n={dazd_dni}): {dazd_priemer:.1f} (-{pokles_dazd_pct:.1f} %) | r={kor_teplota:.2f}")
print(f"Q1 | Marec–október 2025: {priemer_2025:.1f} vs. 2026: {priemer_2026:.1f} ({zmena_yoy_pct:+.1f} %)")

fig, osi = plt.subplots(1, 2, figsize=(13, 5))
osi[0].scatter(kal.loc[~kal["dazdivy"], "temp_avg"], kal.loc[~kal["dazdivy"], "jazdy"], color=MODRA, alpha=0.55, s=25, label=f"Suchý deň (n={sucho_dni})")
osi[0].scatter(kal.loc[kal["dazdivy"], "temp_avg"], kal.loc[kal["dazdivy"], "jazdy"], color=ORANZOVA, alpha=0.65, s=25, label=f"Daždivý deň >= 1 mm (n={dazd_dni})")
osi[0].set_xlabel("Priemerná denná teplota (°C)")
osi[0].set_ylabel("Počet jázd za deň (jazdy)")
osi[0].legend(frameon=False, loc="upper left")
osi[0].set_title(f"V daždivých dňoch klesá počet jázd o {pokles_dazd_pct:.0f} % ({dazd_priemer:.0f} vs. {sucho_priemer:.0f} jázd/deň)", loc="left")

mesiace_popis = ["Mar", "Apr", "Máj", "Jún", "Júl", "Aug", "Sep", "Okt"]
x_pos = np.arange(len(mesiace_popis))
osi[1].bar(x_pos - 0.18, medzirocne[2025].values, width=0.36, color=SVETLE_SEDA, label="2025")
osi[1].bar(x_pos + 0.18, medzirocne[2026].values, width=0.36, color=MODRA, label="2026")
osi[1].set_xticks(x_pos)
osi[1].set_xticklabels(mesiace_popis)
osi[1].set_xlabel("Mesiac (marec–október)")
osi[1].set_ylabel("Priemer jázd za deň (jazdy/deň)")
osi[1].legend(frameon=False, loc="upper left")
osi[1].set_title(f"Medziročne (marec–október) denný dopyt vzrástol o {zmena_yoy_pct:+.1f} % ({priemer_2026:.0f} vs. {priemer_2025:.0f})", loc="left")

fig.tight_layout()
fig.savefig(os.path.join(VYSTUPY, "01_pocasi_a_sezona.png"), dpi=120)
plt.show()

# =====================================================================================
# 4) OTÁZKA 2: KEDY SA JAZDÍ A KTO (HEATMAPY DEŇ × HODINA)
# =====================================================================================
typy_uziv = ["predplatne", "pay_as_you_go", "turista"]
dni_popis = ["Po", "Ut", "St", "Št", "Pi", "So", "Ne"]
pocty_dni = pd.to_datetime(kal["datum"]).dt.day_of_week.value_counts().sort_index()

matice = {}
max_val = 0
for typ in typy_uziv:
    sub = df[df["user_type"] == typ]
    tab = pd.crosstab(sub["den_v_tyzdni"], sub["hodina"]).reindex(index=range(7), columns=range(24), fill_value=0)
    tab = tab.div(pocty_dni, axis=0)
    matice[typ] = tab
    max_val = max(max_val, tab.values.max())

podiel_pred = (df["user_type"] == "predplatne").mean() * 100
spicka_pred = matice["predplatne"].iloc[:5].mean().idxmax()
spicka_tur = matice["turista"].iloc[5:].mean().idxmax()

fig, osi = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
nadpisy_typ = {"predplatne": "Predplatné", "pay_as_you_go": "Pay-as-you-go", "turista": "Turista"}
for ax, typ in zip(osi, typy_uziv):
    sns.heatmap(matice[typ], cmap="Blues", vmin=0, vmax=max_val, cbar=(typ == "turista"),
                cbar_kws={"label": "Priemer jázd / hodinu"} if typ == "turista" else None, ax=ax)
    ax.grid(False)
    ax.set_title(nadpisy_typ[typ], loc="left")
    ax.set_xlabel("Hodina dňa (h)")
    ax.set_yticklabels(dni_popis, rotation=0)
osi[0].set_ylabel("Deň v týždni")
fig.suptitle(f"Predplatné tvorí {podiel_pred:.0f} % jázd so špičkou v pracovné dni o {spicka_pred}:00, turisti jazdia hlavne cez víkend okolo {spicka_tur}:00",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(VYSTUPY, "02_heatmapa_uzivatele.png"), dpi=120)
plt.show()

# =====================================================================================
# 5) OTÁZKA 3: ZDRAVIE VOZOVÉHO PARKU
# =====================================================================================
df["kratka_jazda"] = df["dlzka_min"] <= 2

kola_stat = df.groupby(["bike_id", "bike_type"]).agg(
    celkom_jazd=("trip_id", "count"),
    kratke_jazdy=("kratka_jazda", "sum")
).reset_index()
kola_stat["podil_kratkych_pct"] = kola_stat["kratke_jazdy"] / kola_stat["celkom_jazd"] * 100
assert kola_stat["celkom_jazd"].sum() == len(df)

top15_kola = kola_stat[kola_stat["celkom_jazd"] >= 30].sort_values("podil_kratkych_pct", ascending=False).head(15)
top15_kola.to_csv(os.path.join(VYSTUPY, "03_problemova_kola_top15.csv"), index=False)

podil_platenych = (df["price_czk"] > 0).mean()
priemer_platba = df.loc[df["price_czk"] > 0, "price_czk"].mean()
kratke_bez_trzby = ((df["kratka_jazda"]) & (df["price_czk"] == 0)).sum()
odhad_usle_trzby = kratke_bez_trzby * podil_platenych * priemer_platba

median_podil = kola_stat["podil_kratkych_pct"].median()
najhorsi_bicykel = top15_kola.iloc[0]
print(f"Q3 | Medián: {median_podil:.1f} % | Najhorší: {najhorsi_bicykel['bike_id']} ({najhorsi_bicykel['podil_kratkych_pct']:.1f} %) | Odhad ušlých tržeb: {odhad_usle_trzby:,.0f} Kč")

fig, osi = plt.subplots(1, 2, figsize=(13, 5))
osi[0].hist(df.loc[df["dlzka_min"] <= 45, "dlzka_min"], bins=45, color=MODRA, edgecolor="white")
osi[0].axvspan(0, 2, color=ORANZOVA, alpha=0.35, label="<= 2 min (podozrivé)")
osi[0].set_xlabel("Dĺžka jazdy (minúty)")
osi[0].set_ylabel("Počet jázd (jazdy)")
osi[0].legend(frameon=False, loc="upper right")
osi[0].set_title(f"Do 2 minút skončilo {df['kratka_jazda'].sum()} jázd ({df['kratka_jazda'].mean()*100:.1f} %); odhad ušlých tržieb {odhad_usle_trzby:,.0f} Kč", loc="left")

top15_plot = top15_kola.iloc[::-1]
farby_kola = np.where(top15_plot["bike_type"] == "elektro", ORANZOVA, MODRA)
osi[1].barh(top15_plot["bike_id"], top15_plot["podil_kratkych_pct"], color=farby_kola)
osi[1].axvline(median_podil, color=SEDA, linestyle="--", linewidth=1, label=f"Medián ({median_podil:.1f} %)")
osi[1].set_xlabel("Podiel krátkych jázd <= 2 min (%)")
osi[1].set_ylabel("ID bicykla")
osi[1].legend(frameon=False, loc="lower right")
osi[1].set_title(f"Najhorší bicykel {najhorsi_bicykel['bike_id']} má {najhorsi_bicykel['podil_kratkych_pct']:.1f} % krátkych jázd (modrá = klasický, oranžová = elektro)", loc="left")

fig.tight_layout()
fig.savefig(os.path.join(VYSTUPY, "03_zdravi_kol.png"), dpi=120)
plt.show()

# =====================================================================================
# 6) OTÁZKA 4: ZMENA CENY ODOMYKANIA 1. 6. 2026 (JÚN–OKTÓBER 2025 VS. 2026)
# =====================================================================================
df_jo = df[df["mesiac"].between(6, 10)].copy()
df_jo["skupina"] = np.where(df_jo["user_type"] == "pay_as_you_go", "pay_as_you_go", "ostatní")

porovnanie_ceny = df_jo.groupby(["rok", "skupina"]).agg(jazdy=("trip_id", "count"), trzba=("price_czk", "sum")).unstack(0)
payg_25 = porovnanie_ceny.loc["pay_as_you_go", ("jazdy", 2025)]
payg_26 = porovnanie_ceny.loc["pay_as_you_go", ("jazdy", 2026)]
payg_zmena = (payg_26 / payg_25 - 1) * 100

ost_25 = porovnanie_ceny.loc["ostatní", ("jazdy", 2025)]
ost_26 = porovnanie_ceny.loc["ostatní", ("jazdy", 2026)]
ost_zmena = (ost_26 / ost_25 - 1) * 100

trzba_payg_25 = porovnanie_ceny.loc["pay_as_you_go", ("trzba", 2025)]
trzba_payg_26 = porovnanie_ceny.loc["pay_as_you_go", ("trzba", 2026)]
print(f"Q4 | pay_as_you_go jazdy: {payg_25} -> {payg_26} ({payg_zmena:+.1f} %) | tržby: {trzba_payg_25:,.0f} -> {trzba_payg_26:,.0f} Kč")

mes_trend = df_jo.groupby(["rok", "mesiac", "skupina"])["trip_id"].count().unstack([0, 2])
fig, osi = plt.subplots(1, 2, figsize=(13, 4.8))
mes_jo = ["Jún", "Júl", "Aug", "Sep", "Okt"]
x_jo = np.arange(len(mes_jo))

osi[0].plot(x_jo, mes_trend[(2025, "pay_as_you_go")].values, marker="o", color=SVETLE_SEDA, label="2025")
osi[0].plot(x_jo, mes_trend[(2026, "pay_as_you_go")].values, marker="o", color=ORANZOVA, label="2026 (po zmene ceny)")
osi[0].set_xticks(x_jo)
osi[0].set_xticklabels(mes_jo)
osi[0].set_ylim(0, None)
osi[0].set_xlabel("Mesiac (jún–október)")
osi[0].set_ylabel("Počet jázd za mesiac (jazdy)")
osi[0].legend(frameon=False, loc="upper right")
osi[0].set_title(f"Pay-as-you-go jazdy: {payg_zmena:+.1f} %, ale tržby vzrástli na {trzba_payg_26:,.0f} Kč", loc="left")

osi[1].plot(x_jo, mes_trend[(2025, "ostatní")].values, marker="o", color=SVETLE_SEDA, label="2025")
osi[1].plot(x_jo, mes_trend[(2026, "ostatní")].values, marker="o", color=MODRA, label="2026")
osi[1].set_xticks(x_jo)
osi[1].set_xticklabels(mes_jo)
osi[1].set_ylim(0, None)
osi[1].set_xlabel("Mesiac (jún–október)")
osi[1].set_ylabel("Počet jázd za mesiac (jazdy)")
osi[1].legend(frameon=False, loc="upper right")
osi[1].set_title(f"Ostatní používatelia (jún–október): zmena o {ost_zmena:+.1f} %", loc="left")

fig.tight_layout()
fig.savefig(os.path.join(VYSTUPY, "04_zmena_ceny.png"), dpi=120)
plt.show()

# =====================================================================================
# 7) OTÁZKA 5: STANICE, RANNÝ TOK A NOVÉ VS. STARÉ STANICE
# =====================================================================================
rano = df[(df["user_type"] == "predplatne") & (~df["je_vikend"]) & (df["hodina"].between(6, 9))]
odchody = rano.groupby("start_station_id")["trip_id"].count()
prichody = rano.groupby("end_station_id")["trip_id"].count()

st_tab = stanice.copy()
st_tab["rano_odchody"] = st_tab["station_id"].map(odchody).fillna(0)
st_tab["rano_prichody"] = st_tab["station_id"].map(prichody).fillna(0)
st_tab["cisty_tok"] = st_tab["rano_prichody"] - st_tab["rano_odchody"]
assert np.isclose(st_tab["cisty_tok"].sum(), 0)

df_od_aprila = df[df["start_dt"] >= pd.Timestamp("2026-04-01")]
dni_od_aprila = (pd.Timestamp("2026-10-31") - pd.Timestamp("2026-04-01")).days + 1
starty_apr = df_od_aprila.groupby("start_station_id")["trip_id"].count()
st_tab["odchody_od_aprila"] = st_tab["station_id"].map(starty_apr).fillna(0)
st_tab["denne_odchody"] = st_tab["odchody_od_aprila"] / dni_od_aprila

nove_vs_stare = st_tab.groupby("je_nova")["denne_odchody"].mean()
stare_priemer = nove_vs_stare.loc[False]
nove_priemer = nove_vs_stare.loc[True]
pomer_nove_pct = nove_priemer / stare_priemer * 100

najvacsi_odliv = st_tab.sort_values("cisty_tok").iloc[0]
najvacsi_priliv = st_tab.sort_values("cisty_tok", ascending=False).iloc[0]
st_tab.to_csv(os.path.join(VYSTUPY, "05_stanice_toky.csv"), index=False)

print(f"Q5 | Odliv: {najvacsi_odliv['name']} ({najvacsi_odliv['cisty_tok']:+.0f}) | Príliv: {najvacsi_priliv['name']} ({najvacsi_priliv['cisty_tok']:+.0f})")
print(f"Q5 | Nové: {nove_priemer:.1f} vs. staré: {stare_priemer:.1f} ({pomer_nove_pct:.0f} %)")

fig, osi = plt.subplots(1, 2, figsize=(14, 5.2))
farby_tok = np.where(st_tab["cisty_tok"] >= 0, MODRA, ORANZOVA)
velkosti_bublin = np.abs(st_tab["cisty_tok"]) * 1.8 + 25
osi[0].scatter(st_tab["lon"], st_tab["lat"], s=velkosti_bublin, c=farby_tok, alpha=0.75, edgecolors="white")
for _, r in pd.concat([st_tab.nsmallest(2, "cisty_tok"), st_tab.nlargest(2, "cisty_tok")]).iterrows():
    osi[0].annotate(f"{r['name']} ({r['cisty_tok']:+.0f})", (r["lon"], r["lat"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
osi[0].set_xlabel("Zemepisná dĺžka (°E)")
osi[0].set_ylabel("Zemepisná šírka (°N)")
osi[0].set_title(f"Ranný čistý tok (6–10h): odliv {najvacsi_odliv['name']} ({najvacsi_odliv['cisty_tok']:+.0f}), príliv {najvacsi_priliv['name']} ({najvacsi_priliv['cisty_tok']:+.0f})", loc="left")

stlpce_st = ["Staré stanice (n=52)", "Nové stanice (n=8)"]
hodnoty_st = [stare_priemer, nove_priemer]
osi[1].bar(stlpce_st, hodnoty_st, color=[SVETLE_SEDA, MODRA], width=0.5)
for i, v in enumerate(hodnoty_st):
    osi[1].text(i, v + 0.05, f"{v:.1f}", ha="center", fontweight="bold", color=SEDA)
osi[1].set_ylim(0, max(hodnoty_st) * 1.25)
osi[1].set_ylabel("Priemer odchodov na stanicu za deň (jazdy/deň)")
osi[1].set_title(f"Nové stanice majú {pomer_nove_pct:.0f} % denného výkonu starých staníc ({nove_priemer:.1f} vs. {stare_priemer:.1f})", loc="left")

fig.tight_layout()
fig.savefig(os.path.join(VYSTUPY, "05_stanice_mapa_a_nove.png"), dpi=120)
plt.show()

print("Hotovo, výstupy sú uložené v priečinku outputs/.")