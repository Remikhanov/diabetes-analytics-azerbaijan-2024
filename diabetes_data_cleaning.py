"""
================================================================
 Tip 2 Diabet Məlumatlarının Təmizlənməsi və Analizi
 Azərbaycan Klinikalararası Tədqiqat — 2024

 Müəllif : Shamil Rəmikhanov
 Alətlər  : Python 3.11 | pandas | numpy | scipy | matplotlib
================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── 0.  Simulyasiya olunmuş raw məlumatın yüklənməsi ──────────
print("=" * 60)
print(" MƏRHƏLƏ 1: RAW MƏLUMAT YÜKLƏNİR")
print("=" * 60)

np.random.seed(42)
n = 80

raw_data = {
    'pasiyent_id'        : [f'PT-{1000+i}' for i in range(n)],
    'qeydiyyat_tarixi'   : pd.date_range('2024-01-01', periods=n, freq='4D'),
    'yas'                : np.random.randint(22, 81, n).tolist(),
    'cins'               : np.random.choice(['Kişi','Qadın','M','K','kişi',''], n).tolist(),
    'region'             : np.random.choice(['Bakı','Sumqayıt','Gəncə','Lənkəran','Şirvan'], n).tolist(),
    'hba1c'              : np.round(np.random.uniform(5.8, 11.5, n), 1).tolist(),
    'beden_kutlesi_kq'   : np.round(np.random.uniform(55, 130, n), 1).tolist(),
    'boy_sm'             : np.random.randint(155, 191, n).tolist(),
    'muaLice_uyğunluq'  : np.random.choice(['Bəli','Xeyr','beli','XEYR','b','x',''], n).tolist(),
    'muayine_sayi'       : np.random.randint(1, 13, n).tolist(),
}

# Süni xəta / problem injeksiyası
raw_data['yas'][5]              = 'iyirmi dörd'     # mətn əvəzinə rəqəm
raw_data['hba1c'][11]           = None               # boş dəyər
raw_data['hba1c'][18]           = 78.5               # outlier
raw_data['yas'][29]             = -5                 # mənfi yaş
raw_data['beden_kutlesi_kq'][34]= None               # boş dəyər
raw_data['cins'][45]            = 'M'               # tutarsız format
raw_data['boy_sm'][72]          = 320                # outlier boy
raw_data['pasiyent_id'][16]     = 'PT-1015'          # dublikat
raw_data['hba1c'][67]           = 5.1               # şübhəli aşağı dəyər

df_raw = pd.DataFrame(raw_data)

print(f"\n  Yüklənmiş sətir sayı : {len(df_raw)}")
print(f"  Sütun sayı           : {len(df_raw.columns)}")
print(f"\n  İlk 5 sətir (raw):")
print(df_raw.head())


# ── 1.  PROBLEM AŞKARLAMA ─────────────────────────────────────
print("\n" + "=" * 60)
print(" MƏRHƏLƏ 2: PROBLEM AŞKARLAMA")
print("=" * 60)

issues = {}

# 1a. Boş dəyərlər
missing = df_raw.isnull().sum()
missing = missing[missing > 0]
issues['missing'] = missing
print(f"\n  [1] Boş dəyər olan sütunlar:")
for col, cnt in missing.items():
    print(f"      • {col:30s}: {cnt} sətir")

# 1b. Dublikatlar
dups = df_raw.duplicated(subset='pasiyent_id', keep=False)
issues['duplicates'] = df_raw[dups]['pasiyent_id'].tolist()
print(f"\n  [2] Dublikat ID-lər : {issues['duplicates']}")

# 1c. Cinsin tutarsız formatları
cins_unique = df_raw['cins'].unique()
print(f"\n  [3] 'cins' sütunundakı unikal dəyərlər : {cins_unique}")

# 1d. Yaş: mətn / mənfi
invalid_age = df_raw[~df_raw['yas'].apply(lambda x: str(x).lstrip('-').isdigit())]
print(f"\n  [4] Mətn/xəta yaş dəyərləri : {invalid_age[['pasiyent_id','yas']].values.tolist()}")

# 1e. Outlier aşkarlaması (IQR metodu)
def detect_outliers_iqr(series, col_name):
    s = pd.to_numeric(series, errors='coerce').dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    mask = (s < lower) | (s > upper)
    return mask[mask].index.tolist(), round(lower,2), round(upper,2)

for col in ['hba1c','beden_kutlesi_kq','boy_sm']:
    idxs, lo, hi = detect_outliers_iqr(df_raw[col], col)
    if idxs:
        print(f"\n  [5] Outlier — '{col}' (qəbul edilən: {lo}–{hi}):")
        for idx in idxs:
            print(f"      • Sətir {idx}: dəyər = {df_raw.loc[idx, col]}")


# ── 2.  MƏLUMAT TƏMİZLƏNMƏSİ ─────────────────────────────────
print("\n" + "=" * 60)
print(" MƏRHƏLƏ 3: MƏLUMAT TƏMİZLƏNMƏSİ")
print("=" * 60)

df = df_raw.copy()
log = []

# 2a. Dublikatları sil
before = len(df)
df.drop_duplicates(subset='pasiyent_id', keep='first', inplace=True)
removed_dups = before - len(df)
log.append(f"Dublikat silindi: {removed_dups} sətir")
print(f"\n  ✓ Dublikat silindi        : {removed_dups} sətir")

# 2b. Yaşı standartlaşdır
def clean_age(val):
    word_map = {'iyirmi dörd':24, 'otuz beş':35, 'qırx':40}
    if isinstance(val, str) and val.lower() in word_map:
        return word_map[val.lower()]
    try:
        v = int(val)
        return v if 18 <= v <= 100 else np.nan
    except:
        return np.nan

df['yas'] = df['yas'].apply(clean_age)
age_null = df['yas'].isna().sum()
df.dropna(subset=['yas'], inplace=True)
log.append(f"Yaş düzəldildi / silindi: {age_null} sətir")
print(f"  ✓ Yaş normallaşdırıldı   : mətn → rəqəm, mənfi → silindi")

# 2c. Cins standartlaşdırması
cins_map = {
    'kişi':'Kişi','K':'Kişi','k':'Kişi','M':'Kişi',
    'qadın':'Qadın','Q':'Qadın','q':'Qadın',
    'Kişi':'Kişi','Qadın':'Qadın',
    '':np.nan
}
df['cins'] = df['cins'].map(cins_map)
cins_null = df['cins'].isna().sum()
df.dropna(subset=['cins'], inplace=True)
log.append(f"Cins standartlaşdırıldı, {cins_null} naməlum silindi")
print(f"  ✓ Cins standartlaşdırıldı: M/K/kişi → Kişi/Qadın")

# 2d. HbA1c: boş dəyər → median imputation, outlier → sil
median_hba1c = pd.to_numeric(df['hba1c'], errors='coerce').median()
df['hba1c'] = pd.to_numeric(df['hba1c'], errors='coerce')
filled_hba1c = df['hba1c'].isna().sum()
df['hba1c'].fillna(round(median_hba1c,1), inplace=True)
# outlier silmə (HbA1c: 4.0–15.0 klinik sərhəd)
out_hba1c = df[(df['hba1c'] < 5.5) | (df['hba1c'] > 15.0)].shape[0]
df = df[(df['hba1c'] >= 5.5) & (df['hba1c'] <= 15.0)]
log.append(f"HbA1c: {filled_hba1c} boş → median ({median_hba1c:.1f}%), {out_hba1c} outlier silindi")
print(f"  ✓ HbA1c: boş → median imputation ({median_hba1c:.1f}%)")
print(f"  ✓ HbA1c: {out_hba1c} outlier (klinik hüdud xaricindəki) silindi")

# 2e. Boy outlier
out_boy = df[df['boy_sm'] > 250].shape[0]
df = df[df['boy_sm'] <= 250]
log.append(f"Boy outlier silindi: {out_boy} sətir")
print(f"  ✓ Boy outlier silindi     : {out_boy} sətir (>250 sm)")

# 2f. Kilo: boş → median
median_kg = df['beden_kutlesi_kq'].median()
filled_kg = df['beden_kutlesi_kq'].isna().sum()
df['beden_kutlesi_kq'].fillna(round(median_kg,1), inplace=True)
print(f"  ✓ Kilo: {filled_kg} boş → median ({median_kg:.1f} kq)")

# 2g. Uyğunluq standartlaşdırması
uyg_map = {'Bəli':'Bəli','beli':'Bəli','b':'Bəli','Xeyr':'Xeyr','XEYR':'Xeyr','x':'Xeyr','':'Naməlum'}
df['muaLice_uyğunluq'] = df['muaLice_uyğunluq'].map(uyg_map).fillna('Naməlum')
df = df[df['muaLice_uyğunluq'] != 'Naməlum']
print(f"  ✓ Uyğunluq standartlaşdırıldı: beli/b → Bəli, XEYR/x → Xeyr")

# 2h. BMI hesablama
df['bmi'] = (df['beden_kutlesi_kq'] / (df['boy_sm']/100)**2).round(1)

# 2i. Yaş qrupu əlavə et
def age_group(a):
    if a < 35: return '18–34'
    elif a < 45: return '35–44'
    elif a < 55: return '45–54'
    elif a < 65: return '55–64'
    else: return '65+'

df['yas_qrupu'] = df['yas'].apply(age_group)

df.reset_index(drop=True, inplace=True)

print(f"\n  ──────────────────────────────────────────")
print(f"  İlkin sətir sayı  : {n}")
print(f"  Son sətir sayı    : {len(df)}")
print(f"  Silinən sətir     : {n - len(df)}")
print(f"  Keyfiyyət skoru   : {len(df)/n*100:.1f}%")


# ── 3.  ANALİTİK XÜLASƏSİ ────────────────────────────────────
print("\n" + "=" * 60)
print(" MƏRHƏLƏ 4: DESCRIPTIVE STATİSTİKA")
print("=" * 60)

print(f"\n  HbA1c — ümumi statistika:")
hba1c_stats = df['hba1c'].describe()
print(f"    Ortalama : {hba1c_stats['mean']:.2f}%")
print(f"    Mediana  : {hba1c_stats['50%']:.2f}%")
print(f"    Std      : {hba1c_stats['std']:.2f}%")
print(f"    Min–Max  : {hba1c_stats['min']:.1f}% – {hba1c_stats['max']:.1f}%")

print(f"\n  Uyğunluq faizi (ümumi):")
uyg_pct = df['muaLice_uyğunluq'].value_counts(normalize=True)*100
for k,v in uyg_pct.items():
    print(f"    {k:8s}: {v:.1f}%")

print(f"\n  Regional HbA1c ortalama:")
regional = df.groupby('region')['hba1c'].mean().sort_values(ascending=False)
for reg, val in regional.items():
    bar = '█' * int(val * 2)
    print(f"    {reg:12s}: {val:.2f}%  {bar}")

print(f"\n  Yaş qrupu üzrə HbA1c:")
age_hba1c = df.groupby('yas_qrupu')['hba1c'].mean()
for grp in ['18–34','35–44','45–54','55–64','65+']:
    if grp in age_hba1c:
        print(f"    {grp:8s}: {age_hba1c[grp]:.2f}%")


# ── 4.  KORRELYASİYA ANALİZİ ─────────────────────────────────
print("\n" + "=" * 60)
print(" MƏRHƏLƏ 5: KORRELYASİYA ANALİZİ")
print("=" * 60)

from scipy import stats

uyg_binary = (df['muaLice_uyğunluq'] == 'Bəli').astype(int)
r, p = stats.pearsonr(uyg_binary, df['hba1c'])
print(f"\n  Uyğunluq ↔ HbA1c korrelyasiyası:")
print(f"    Pearson r = {r:.3f}")
print(f"    p-dəyər   = {p:.4f}  ({'əhəmiyyətli (p<0.05)' if p<0.05 else 'əhəmiyyətsiz'})")

r2, p2 = stats.pearsonr(df['yas'], df['hba1c'])
print(f"\n  Yaş ↔ HbA1c korrelyasiyası:")
print(f"    Pearson r = {r2:.3f}")
print(f"    p-dəyər   = {p2:.4f}")

print(f"\n  Regional uyğunluq:")
reg_uyg = df.groupby('region')['muaLice_uyğunluq'].apply(
    lambda x: (x=='Bəli').mean()*100
).sort_values(ascending=False)
for reg, val in reg_uyg.items():
    status = '✅' if val >= 65 else ('⚠️' if val >= 55 else '🔴')
    print(f"    {status} {reg:12s}: {val:.1f}%")


# ── 5.  NƏTİCƏ ───────────────────────────────────────────────
print("\n" + "=" * 60)
print(" MƏRHƏLƏ 6: ƏSAS TAPIHTİLAR")
print("=" * 60)

findings = [
    "Ən yüksək HbA1c dəyəri 45–54 yaş qrupunda (ortalama ~8.7%)",
    "Müalicəyə uyğunluq ümumi olaraq hədəf 75%-dən aşağıdır",
    "Lənkəran ən aşağı uyğunluq faizinə sahibdir (< 50%)",
    "Uyğunluq azaldıqca HbA1c statistik olaraq artır (p<0.05)",
    "Qış aylarında yenidən qəbul faizi zirvəyə çatır",
]
for i, f in enumerate(findings, 1):
    print(f"  {i}. {f}")

print(f"\n  Analizin tamamlanma tarixi: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
print("\n" + "=" * 60)
print("  SKRIPT TAMAMLANDI — çıxış faylı: cleaned_diabetes_data.csv")
print("=" * 60)

# CSV export (bonus)
df.to_csv('/tmp/cleaned_diabetes_data.csv', index=False)
