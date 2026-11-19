# =====================================================================================
# ZÁVEREČNÝ PROJEKT A: OstraBike (1. 3. 2025 – 31. 10. 2026)
# Otázka: Jak se systému OstraBike daří po migraci dat, co ovlivňuje poptávku a stav
# kol, jak dopadla změna ceny odemykání 1. 6. 2026 a kam směřovat investice?
#
# Závěr (3 věty):
# 1. Po vyčištění 40 000 řádků zůstalo 39 196 platných jízd (vyřazeno 2,0 %) a denní
#    poptávka silně souvisí s teplotou (r = 0,85) i deštěm (v deštivých dnech >= 1 mm
#    klesá průměr o 43,8 % na 40,3 jízd/den oproti 71,7 jízd/den v suchých dnech),
#    zatímco sezóna březen–říjen meziročně vzrostla o +16,1 % (z 68,5 na 79,6 jízd/den).
# 2. Změna ceny odemykání od 1. 6. 2026 snížila v období červen–říjen jízdy
#    pay_as_you_go o -6,0 % (z 3 954 na 3 715), ale tržby této skupiny vzrostly ze
#    129 243 Kč na 140 920 Kč (ostatní jízdy vzrostly o +21,9 %), a u kol se
#    ukázal problém u konkrétních kusů (medián krátkých jízd <= 2 min je 0,0 %,
#    nejhorší kolo B0015 má 99,6 %, odhad ušlých přímých tržeb je 46 322 Kč).
# 3. Provozní tým by měl prioritně opravit nebo vyřadit problémová kola s vysokým
#    podílem krátkých jízd, posílit ranní přesun kol ze stanic s odlivem (např.
#    Fakultní nemocnice, -251 jízd) do lokalit s přílivem (např. Nová Ves, +183 jízd)
#    a dále rozvíjet 8 nových stanic z dubna 2026, které mají 122 % denního výkonu
#    starých stanic (1,7 vs. 1,4 odjezdů/den).
# =====================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cesta k datům a vytvoření složky outputs pro grafy a tabulky
CESTA = r"C:\Users\matej\Downloads\A_OstraBike\A_OstraBike"
os.makedirs("outputs", exist_ok=True)

# Základní barvy a nastavení grafů z hodin
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
# 1) NAČTENÍ DAT A ČIŠTĚNÍ S LOGEM
# =====================================================================================
jizdy_raw = pd.read_csv(os.path.join(CESTA, "jizdy.csv"))
stanice_raw = pd.read_csv(os.path.join(CESTA, "stanice.csv"))
kola_raw = pd.read_csv(os.path.join(CESTA, "kola.csv"))
uzivatele_raw = pd.read_csv(os.path.join(CESTA, "uzivatele.csv"))
pocasi_raw = pd.read_csv(os.path.join(CESTA, "pocasi.csv"))

print("Řádky raw:", len(jizdy_raw), len(stanice_raw), len(kola_raw), len(uzivatele_raw), len(pocasi_raw))

log_cisteni = []
df = jizdy_raw.copy()
puvodni_pocet = len(df)

# Duplicitní řádky: ponecháme první výskyt, duplicity by zkreslily počty jízd i tržby
dup_cele = df.duplicated().sum()
df = df.drop_duplicates()
dup_id = df.duplicated(subset=["trip_id"]).sum()
df = df.drop_duplicates(subset=["trip_id"], keep="first")
log_cisteni.append(["Duplicity (celé + trip_id)", int(dup_cele + dup_id), len(df)])
print("Duplicity odstraněny:", dup_cele + dup_id, "-> zbývá:", len(df))

# Dva formáty času: nový yyyy-mm-dd hh:mm a starý dd.mm.yyyy hh:mm převedeme na datum a čas
start_iso = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M", errors="coerce")
start_stary = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M", errors="coerce")
df["start_dt"] = start_iso.fillna(start_stary)

end_iso = pd.to_datetime(df["end_time"], format="%Y-%m-%d %H:%M", errors="coerce")
end_stary = pd.to_datetime(df["end_time"], format="%d.%m.%Y %H:%M", errors="coerce")
df["end_dt"] = end_iso.fillna(end_stary)

pocet_stary_cas = int(start_stary.notna().sum())
df = df.dropna(subset=["start_dt", "end_dt"]).copy()
log_cisteni.append(["Dva formáty času (sjednoceno)", pocet_stary_cas, len(df)])
print("Starý formát času sjednocen:", pocet_stary_cas)

df["delka_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60

# Konec před začátkem: záporná délka jízdy je chyba záznamu, vyřadíme ji
pred_zacatkem = int((df["delka_min"] < 0).sum())
df = df[df["delka_min"] >= 0].copy()
log_cisteni.append(["Konec před začátkem", pred_zacatkem, len(df)])
print("Vyřazeno konec před začátkem:", pred_zacatkem)

# Jízdy delší než 24 hodin (1440 min): neuzavřené/extrémní výpůjčky vyřadíme, aby nekazily průměr
nad_24h = int((df["delka_min"] > 1440).sum())
df = df[df["delka_min"] <= 1440].copy()
log_cisteni.append(["Jízdy delší než 24 h", nad_24h, len(df)])
print("Vyřazeno > 24 h:", nad_24h)

# distance_km jako text s čárkou: čárku změníme na tečku a převedeme na číslo
carky_km = int(df["distance_km"].astype(str).str.contains(",").sum())
df["distance_km"] = pd.to_numeric(df["distance_km"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
df = df[df["distance_km"] >= 0].copy()
log_cisteni.append(["distance_km (čárka -> číslo)", carky_km, len(df)])
print("Opraveno distance_km s čárkou:", carky_km)

# Malá písmena v ID: sjednotíme na velká písmena, aby párování tabulek fungovalo
df["start_station_id"] = df["start_station_id"].astype(str).str.strip().str.upper()
df["end_station_id"] = df["end_station_id"].astype(str).str.strip().str.upper()
df["bike_id"] = df["bike_id"].astype(str).str.strip().str.upper()

# Stanice mimo číselník (např. ST999): vyřadíme, protože chybí v číselníku stanic
platne_stanice = set(stanice_raw["station_id"].astype(str).str.strip().str.upper())
mimo_stanice = (~df["start_station_id"].isin(platne_stanice)) | (~df["end_station_id"].isin(platne_stanice))
print("Vyřazeno stanice mimo číselník:", int(mimo_stanice.sum()))
df = df[~mimo_stanice].copy()
log_cisteni.append(["Stanice mimo číselník", int(mimo_stanice.sum()), len(df)])

# Záporné ceny: dobropisy/storna vyřadíme, aby nekazily kladnou tržbu
zaporne_ceny = int((df["price_czk"] < 0).sum())
print("Vyřazeno záporné ceny:", zaporne_ceny)
df = df[df["price_czk"] >= 0].copy()
log_cisteni.append(["Záporné ceny", zaporne_ceny, len(df)])

# Prázdné user_id: ověříme, že je jen u turistů (vlastnost dat, ne chyba)
print("Prázdné user_id podle user_type:")
print(df.groupby("user_type")["user_id"].apply(lambda s: s.isna().sum()))

# Počasí a uživatelé: chybějící teplotu doplníme interpolací, nesmyslné roky narození dáme na NaN
pocasi = pocasi_raw.copy()
pocasi["datum"] = pd.to_datetime(pocasi["date"]).dt.date
chybi_teplota = int(pocasi["temp_avg"].isna().sum())
pocasi["temp_avg"] = pocasi["temp_avg"].interpolate(method="linear").bfill().ffill()
print("Doplněno teplot v počasí:", chybi_teplota)

uzivatele = uzivatele_raw.copy()
uzivatele["user_id"] = uzivatele["user_id"].astype(str).str.strip().str.upper()
rok_nar = np.floor(pd.to_numeric(uzivatele["birth_year"], errors="coerce"))
spatne_roky = int(((rok_nar < 1935) | (rok_nar > 2012)).sum())
rok_nar[(rok_nar < 1935) | (rok_nar > 2012)] = np.nan
uzivatele["rok_narozeni"] = rok_nar
print("Vyčištěno nesmyslných roků narození:", spatne_roky)

log_df = pd.DataFrame(log_cisteni, columns=["krok", "pocet_zmen", "zbyva_radku"])
log_df.to_csv("outputs/00_log_cisteni.csv", index=False)
print("Čistých jízd:", len(df), "z", puvodni_pocet)

# =====================================================================================
# 2) SPOJENÍ TABULEK A KONTROLY ASSERT
# =====================================================================================
stanice = stanice_raw.copy()
stanice["station_id"] = stanice["station_id"].astype(str).str.strip().str.upper()
stanice["je_nova"] = pd.to_datetime(stanice["opened"]) >= pd.Timestamp("2026-04-01")

kola = kola_raw.copy()
kola["bike_id"] = kola["bike_id"].astype(str).str.strip().str.upper()

df["datum"] = df["start_dt"].dt.date
df["rok"] = df["start_dt"].dt.year
df["mesic"] = df["start_dt"].dt.month
df["hodina"] = df["start_dt"].dt.hour
df["den_v_tydnu"] = df["start_dt"].dt.day_of_week
df["je_vikend"] = df["den_v_tydnu"] >= 5

n_pred_merge = len(df)
trzba_pred_merge = df["price_czk"].sum()

df = df.merge(
    stanice[["station_id", "name", "district", "station_type", "lat", "lon", "capacity", "je_nova"]],
    left_on="start_station_id", right_on="station_id", how="left"
).rename(columns={"name": "start_nazev", "district": "start_cast", "station_type": "start_typ", "je_nova": "start_nova"}).drop(columns=["station_id"])
assert len(df) == n_pred_merge

df = df.merge(
    stanice[["station_id", "name", "je_nova"]],
    left_on="end_station_id", right_on="station_id", how="left"
).rename(columns={"name": "cil_nazev", "je_nova": "cil_nova"}).drop(columns=["station_id"])
assert len(df) == n_pred_merge

df = df.merge(kola[["bike_id", "bike_type"]], on="bike_id", how="left")
assert len(df) == n_pred_merge

df = df.merge(pocasi[["datum", "temp_avg", "rain_mm", "wind_ms"]], on="datum", how="left")
assert len(df) == n_pred_merge
assert df["temp_avg"].isna().sum() == 0

df["uid_ciste"] = df["user_id"].fillna("").astype(str).str.strip().str.upper()
df = df.merge(uzivatele[["user_id", "rok_narozeni"]], left_on="uid_ciste", right_on="user_id", how="left", suffixes=("", "_u")).drop(columns=["user_id_u", "uid_ciste"])
assert len(df) == n_pred_merge
assert np.isclose(df["price_czk"].sum(), trzba_pred_merge)

# =====================================================================================
# 3) OTÁZKA 1: SEZÓNNÍ VÝVOJ A POČASÍ
# =====================================================================================
denni = df.groupby("datum").agg(jizdy=("trip_id", "count"), trzba_kc=("price_czk", "sum")).reset_index()
kal = pocasi.merge(denni, on="datum", how="left")
kal["jizdy"] = kal["jizdy"].fillna(0)
kal["trzba_kc"] = kal["trzba_kc"].fillna(0)
assert len(kal) == len(pocasi)
assert kal["jizdy"].sum() == len(df)

kal["rok"] = pd.to_datetime(kal["datum"]).dt.year
kal["mesic"] = pd.to_datetime(kal["datum"]).dt.month
kal["destivy"] = kal["rain_mm"] >= 1.0

po_desti = kal.groupby("destivy")["jizdy"].agg(["count", "mean"])
sucho_prumer = po_desti.loc[False, "mean"]
dest_prumer = po_desti.loc[True, "mean"]
sucho_dni = int(po_desti.loc[False, "count"])
dest_dni = int(po_desti.loc[True, "count"])
pokles_dest_pct = (1 - dest_prumer / sucho_prumer) * 100
kor_teplota = kal["temp_avg"].corr(kal["jizdy"])

meziletni = kal[kal["mesic"].between(3, 10)].groupby(["rok", "mesic"])["jizdy"].mean().unstack(0)
prumer_2025 = kal[(kal["mesic"].between(3, 10)) & (kal["rok"] == 2025)]["jizdy"].mean()
prumer_2026 = kal[(kal["mesic"].between(3, 10)) & (kal["rok"] == 2026)]["jizdy"].mean()
zmena_yoy_pct = (prumer_2026 / prumer_2025 - 1) * 100

print(f"Q1 | Sucho (n={sucho_dni}): {sucho_prumer:.1f} | Déšť (n={dest_dni}): {dest_prumer:.1f} (-{pokles_dest_pct:.1f} %) | r={kor_teplota:.2f}")
print(f"Q1 | Březen–říjen 2025: {prumer_2025:.1f} vs. 2026: {prumer_2026:.1f} ({zmena_yoy_pct:+.1f} %)")

fig, osy = plt.subplots(1, 2, figsize=(13, 5))
osy[0].scatter(kal.loc[~kal["destivy"], "temp_avg"], kal.loc[~kal["destivy"], "jizdy"], color=MODRA, alpha=0.55, s=25, label=f"Suchý den (n={sucho_dni})")
osy[0].scatter(kal.loc[kal["destivy"], "temp_avg"], kal.loc[kal["destivy"], "jizdy"], color=ORANZOVA, alpha=0.65, s=25, label=f"Deštivý den >= 1 mm (n={dest_dni})")
osy[0].set_xlabel("Průměrná denní teplota (°C)")
osy[0].set_ylabel("Počet jízd za den (jízdy)")
osy[0].legend(frameon=False, loc="upper left")
osy[0].set_title(f"V deštivých dnech klesá počet jízd o {pokles_dest_pct:.0f} % ({dest_prumer:.0f} vs. {sucho_prumer:.0f} jízd/den)", loc="left")

mesice_popis = ["Bře", "Dub", "Kvě", "Čer", "Čvc", "Srp", "Zář", "Říj"]
x_pos = np.arange(len(mesice_popis))
osy[1].bar(x_pos - 0.18, meziletni[2025].values, width=0.36, color=SVETLE_SEDA, label="2025")
osy[1].bar(x_pos + 0.18, meziletni[2026].values, width=0.36, color=MODRA, label="2026")
osy[1].set_xticks(x_pos)
osy[1].set_xticklabels(mesice_popis)
osy[1].set_xlabel("Měsíc (březen–říjen)")
osy[1].set_ylabel("Průměr jízd za den (jízdy/den)")
osy[1].legend(frameon=False, loc="upper left")
osy[1].set_title(f"Meziročně (březen–říjen) denní poptávka vzrostla o {zmena_yoy_pct:+.1f} % ({prumer_2026:.0f} vs. {prumer_2025:.0f})", loc="left")

fig.tight_layout()
fig.savefig("outputs/01_pocasi_a_sezona.png", dpi=120)
plt.show()

# =====================================================================================
# 4) OTÁZKA 2: KDY SE JEZDÍ A KDO (HEATMAPY DEN × HODINA)
# =====================================================================================
typy_uziv = ["predplatne", "pay_as_you_go", "turista"]
dny_popis = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"]
pocty_dnu = pd.to_datetime(kal["datum"]).dt.day_of_week.value_counts().sort_index()

matice = {}
max_val = 0
for typ in typy_uziv:
    sub = df[df["user_type"] == typ]
    tab = pd.crosstab(sub["den_v_tydnu"], sub["hodina"]).reindex(index=range(7), columns=range(24), fill_value=0)
    tab = tab.div(pocty_dnu, axis=0)
    matice[typ] = tab
    max_val = max(max_val, tab.values.max())

podil_pred = (df["user_type"] == "predplatne").mean() * 100
spicka_pred = matice["predplatne"].iloc[:5].mean().idxmax()
spicka_tur = matice["turista"].iloc[5:].mean().idxmax()

fig, osy = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
nadpisy_typ = {"predplatne": "Předplatné", "pay_as_you_go": "Pay-as-you-go", "turista": "Turista"}
for ax, typ in zip(osy, typy_uziv):
    sns.heatmap(matice[typ], cmap="Blues", vmin=0, vmax=max_val, cbar=(typ == "turista"),
                cbar_kws={"label": "Průměr jízd / hodinu"} if typ == "turista" else None, ax=ax)
    ax.grid(False)
    ax.set_title(nadpisy_typ[typ], loc="left")
    ax.set_xlabel("Hodina dne (h)")
    ax.set_yticklabels(dny_popis, rotation=0)
osy[0].set_ylabel("Den v týdnu")
fig.suptitle(f"Předplatné tvoří {podil_pred:.0f} % jízd se špičkou v pracovní dny v {spicka_pred}:00, turisté jezdí hlavně o víkendu kolem {spicka_tur}:00",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/02_heatmapa_uzivatele.png", dpi=120)
plt.show()

# =====================================================================================
# 5) OTÁZKA 3: ZDRAVÍ VOZOVÉHO PARKU
# =====================================================================================
df["kratka_jizda"] = df["delka_min"] <= 2

kola_stat = df.groupby(["bike_id", "bike_type"]).agg(
    celkem_jizd=("trip_id", "count"),
    kratke_jizdy=("kratka_jizda", "sum")
).reset_index()
kola_stat["podil_kratkych_pct"] = kola_stat["kratke_jizdy"] / kola_stat["celkem_jizd"] * 100
assert kola_stat["celkem_jizd"].sum() == len(df)

top15_kola = kola_stat[kola_stat["celkem_jizd"] >= 30].sort_values("podil_kratkych_pct", ascending=False).head(15)
top15_kola.to_csv("outputs/03_problemova_kola_top15.csv", index=False)

podil_placenych = (df["price_czk"] > 0).mean()
prumer_platba = df.loc[df["price_czk"] > 0, "price_czk"].mean()
kratke_bez_trzby = ((df["kratka_jizda"]) & (df["price_czk"] == 0)).sum()
odhad_usle_trzby = kratke_bez_trzby * podil_placenych * prumer_platba

median_podil = kola_stat["podil_kratkych_pct"].median()
nejhorsi_kolo = top15_kola.iloc[0]
print(f"Q3 | Medián: {median_podil:.1f} % | Nejhorší: {nejhorsi_kolo['bike_id']} ({nejhorsi_kolo['podil_kratkych_pct']:.1f} %) | Odhad ušlých tržeb: {odhad_usle_trzby:,.0f} Kč")

fig, osy = plt.subplots(1, 2, figsize=(13, 5))
osy[0].hist(df.loc[df["delka_min"] <= 45, "delka_min"], bins=45, color=MODRA, edgecolor="white")
osy[0].axvspan(0, 2, color=ORANZOVA, alpha=0.35, label="<= 2 min (podezřelé)")
osy[0].set_xlabel("Délka jízdy (minuty)")
osy[0].set_ylabel("Počet jízd (jízdy)")
osy[0].legend(frameon=False, loc="upper right")
osy[0].set_title(f"Do 2 minut skončilo {df['kratka_jizda'].sum()} jízd ({df['kratka_jizda'].mean()*100:.1f} %); odhad ušlých tržeb {odhad_usle_trzby:,.0f} Kč", loc="left")

top15_plot = top15_kola.iloc[::-1]
barvy_kola = np.where(top15_plot["bike_type"] == "elektro", ORANZOVA, MODRA)
osy[1].barh(top15_plot["bike_id"], top15_plot["podil_kratkych_pct"], color=barvy_kola)
osy[1].axvline(median_podil, color=SEDA, linestyle="--", linewidth=1, label=f"Medián ({median_podil:.1f} %)")
osy[1].set_xlabel("Podíl krátkých jízd <= 2 min (%)")
osy[1].set_ylabel("ID kola")
osy[1].legend(frameon=False, loc="lower right")
osy[1].set_title(f"Nejhorší kolo {nejhorsi_kolo['bike_id']} má {nejhorsi_kolo['podil_kratkych_pct']:.1f} % krátkých jízd (modrá = klasické, oranžová = elektro)", loc="left")

fig.tight_layout()
fig.savefig("outputs/03_zdravi_kol.png", dpi=120)
plt.show()

# =====================================================================================
# 6) OTÁZKA 4: ZMĚNA CENY ODEMYKÁNÍ 1. 6. 2026 (ČERVEN–ŘÍJEN 2025 VS. 2026)
# =====================================================================================
df_cr = df[df["mesic"].between(6, 10)].copy()
df_cr["skupina"] = np.where(df_cr["user_type"] == "pay_as_you_go", "pay_as_you_go", "ostatní")

srovnani_ceny = df_cr.groupby(["rok", "skupina"]).agg(jizdy=("trip_id", "count"), trzba=("price_czk", "sum")).unstack(0)
payg_25 = srovnani_ceny.loc["pay_as_you_go", ("jizdy", 2025)]
payg_26 = srovnani_ceny.loc["pay_as_you_go", ("jizdy", 2026)]
payg_zmena = (payg_26 / payg_25 - 1) * 100

ost_25 = srovnani_ceny.loc["ostatní", ("jizdy", 2025)]
ost_26 = srovnani_ceny.loc["ostatní", ("jizdy", 2026)]
ost_zmena = (ost_26 / ost_25 - 1) * 100

trzba_payg_25 = srovnani_ceny.loc["pay_as_you_go", ("trzba", 2025)]
trzba_payg_26 = srovnani_ceny.loc["pay_as_you_go", ("trzba", 2026)]
print(f"Q4 | pay_as_you_go jízdy: {payg_25} -> {payg_26} ({payg_zmena:+.1f} %) | tržby: {trzba_payg_25:,.0f} -> {trzba_payg_26:,.0f} Kč")

mes_trend = df_cr.groupby(["rok", "mesic", "skupina"])["trip_id"].count().unstack([0, 2])
fig, osy = plt.subplots(1, 2, figsize=(13, 4.8))
mes_cr = ["Čer", "Čvc", "Srp", "Zář", "Říj"]
x_cr = np.arange(len(mes_cr))

osy[0].plot(x_cr, mes_trend[(2025, "pay_as_you_go")].values, marker="o", color=SVETLE_SEDA, label="2025")
osy[0].plot(x_cr, mes_trend[(2026, "pay_as_you_go")].values, marker="o", color=ORANZOVA, label="2026 (po změně ceny)")
osy[0].set_xticks(x_cr)
osy[0].set_xticklabels(mes_cr)
osy[0].set_ylim(0, None)
osy[0].set_xlabel("Měsíc (červen–říjen)")
osy[0].set_ylabel("Počet jízd za měsíc (jízdy)")
osy[0].legend(frameon=False, loc="upper right")
osy[0].set_title(f"Pay-as-you-go jízdy: {payg_zmena:+.1f} %, ale tržby vzrostly na {trzba_payg_26:,.0f} Kč", loc="left")

osy[1].plot(x_cr, mes_trend[(2025, "ostatní")].values, marker="o", color=SVETLE_SEDA, label="2025")
osy[1].plot(x_cr, mes_trend[(2026, "ostatní")].values, marker="o", color=MODRA, label="2026")
osy[1].set_xticks(x_cr)
osy[1].set_xticklabels(mes_cr)
osy[1].set_ylim(0, None)
osy[1].set_xlabel("Měsíc (červen–říjen)")
osy[1].set_ylabel("Počet jízd za měsíc (jízdy)")
osy[1].legend(frameon=False, loc="upper right")
osy[1].set_title(f"Ostatní uživatelé (červen–říjen): změna o {ost_zmena:+.1f} %", loc="left")

fig.tight_layout()
fig.savefig("outputs/04_zmena_ceny.png", dpi=120)
plt.show()

# =====================================================================================
# 7) OTÁZKA 5: STANICE, RANNÍ TOK A NOVÉ VS. STARÉ STANICE
# =====================================================================================
rano = df[(df["user_type"] == "predplatne") & (~df["je_vikend"]) & (df["hodina"].between(6, 9))]
odjezdy = rano.groupby("start_station_id")["trip_id"].count()
prijezdy = rano.groupby("end_station_id")["trip_id"].count()

st_tab = stanice.copy()
st_tab["rano_odjezdy"] = st_tab["station_id"].map(odjezdy).fillna(0)
st_tab["rano_prijezdy"] = st_tab["station_id"].map(prijezdy).fillna(0)
st_tab["cisty_tok"] = st_tab["rano_prijezdy"] - st_tab["rano_odjezdy"]
assert np.isclose(st_tab["cisty_tok"].sum(), 0)

df_od_dubna = df[df["start_dt"] >= pd.Timestamp("2026-04-01")]
dni_od_dubna = (pd.Timestamp("2026-10-31") - pd.Timestamp("2026-04-01")).days + 1
starty_dub = df_od_dubna.groupby("start_station_id")["trip_id"].count()
st_tab["odjezdy_od_dubna"] = st_tab["station_id"].map(starty_dub).fillna(0)
st_tab["denni_odjezdy"] = st_tab["odjezdy_od_dubna"] / dni_od_dubna

nove_vs_stare = st_tab.groupby("je_nova")["denni_odjezdy"].mean()
stare_prumer = nove_vs_stare.loc[False]
nove_prumer = nove_vs_stare.loc[True]
pomer_nove_pct = nove_prumer / stare_prumer * 100

nejvetsi_odliv = st_tab.sort_values("cisty_tok").iloc[0]
nejvetsi_priliv = st_tab.sort_values("cisty_tok", ascending=False).iloc[0]
st_tab.to_csv("outputs/05_stanice_toky.csv", index=False)

print(f"Q5 | Odliv: {nejvetsi_odliv['name']} ({nejvetsi_odliv['cisty_tok']:+.0f}) | Příliv: {nejvetsi_priliv['name']} ({nejvetsi_priliv['cisty_tok']:+.0f})")
print(f"Q5 | Nové: {nove_prumer:.1f} vs. staré: {stare_prumer:.1f} ({pomer_nove_pct:.0f} %)")

fig, osy = plt.subplots(1, 2, figsize=(14, 5.2))
barvy_tok = np.where(st_tab["cisty_tok"] >= 0, MODRA, ORANZOVA)
velikosti_bublin = np.abs(st_tab["cisty_tok"]) * 1.8 + 25
osy[0].scatter(st_tab["lon"], st_tab["lat"], s=velikosti_bublin, c=barvy_tok, alpha=0.75, edgecolors="white")
for _, r in pd.concat([st_tab.nsmallest(2, "cisty_tok"), st_tab.nlargest(2, "cisty_tok")]).iterrows():
    osy[0].annotate(f"{r['name']} ({r['cisty_tok']:+.0f})", (r["lon"], r["lat"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
osy[0].set_xlabel("Zeměpisná délka (°E)")
osy[0].set_ylabel("Zeměpisná šířka (°N)")
osy[0].set_title(f"Ranní čistý tok (6–10h): odliv {nejvetsi_odliv['name']} ({nejvetsi_odliv['cisty_tok']:+.0f}), příliv {nejvetsi_priliv['name']} ({nejvetsi_priliv['cisty_tok']:+.0f})", loc="left")

sloupce_st = ["Staré stanice (n=52)", "Nové stanice (n=8)"]
hodnoty_st = [stare_prumer, nove_prumer]
osy[1].bar(sloupce_st, hodnoty_st, color=[SVETLE_SEDA, MODRA], width=0.5)
for i, v in enumerate(hodnoty_st):
    osy[1].text(i, v + 0.05, f"{v:.1f}", ha="center", fontweight="bold", color=SEDA)
osy[1].set_ylim(0, max(hodnoty_st) * 1.25)
osy[1].set_ylabel("Průměr odjezdů na stanici za den (jízdy/den)")
osy[1].set_title(f"Nové stanice mají {pomer_nove_pct:.0f} % denního výkonu starých stanic ({nove_prumer:.1f} vs. {stare_prumer:.1f})", loc="left")

fig.tight_layout()
fig.savefig("outputs/05_stanice_mapa_a_nove.png", dpi=120)
plt.show()

print("Hotovo, výstupy jsou uloženy ve složce outputs/.")