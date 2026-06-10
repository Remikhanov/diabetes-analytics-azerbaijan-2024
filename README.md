# 🏥 Tip 2 Diabet Analizi — Azərbaycan 2024

> Azərbaycan üzrə 5 regiondan 3,820 pasiyent məlumatının 
> data cleaning, EDA və regional müqayisəli analizi.

## 📌 Layihə Haqqında

Bu layihədə klinikalardan gəlmiş xam (raw) EHR məlumatları 
Python ilə təmizlənmiş, analiz edilmiş və vizuallaşdırılmışdır.

## 🔍 Əsas Tapıntılar

- 45–54 yaş qrupu ən yüksək HbA1c dəyərinə sahibdir (8.7%)
- Müalicəyə uyğunluq hədəf 75%-dən aşağıdır (61.4%)
- Lənkəran kritik bölgə: uyğunluq 49%, yenidən qəbul 31%
- Uyğunluq ↔ yenidən qəbul: r = −0.84 (güclü korrelyasiya)

## 🛠️ İstifadə Edilən Alətlər

| Alət | Məqsəd |
|------|--------|
| Python 3.11 | Əsas analiz dili |
| pandas | Data cleaning & manipulation |
| numpy | Statistik hesablamalar |
| scipy | Korrelyasiya analizi (Pearson r) |
| Power BI | İnteraktiv dashboard |

## 📁 Fayl Strukturu
├── diabetes_data_cleaning.py   # Data cleaning pipeline
├── Diabetes_Data.xlsx          # Raw + Clean dataset (3 sheet)
└── README.md

## 🔄 Data Cleaning Prosesi

| Problem | Həll |
|---------|------|
| Mətn formatında yaş | word-to-int mapping |
| Boş HbA1c dəyəri | Median imputation |
| Outlier (HbA1c=78.5%) | IQR filter + klinik hüdud |
| Dublikat pasiyent | drop_duplicates() |
| Cins tutarsızlığı | Regex + map() |

**Nəticə:** 80 → 76 sətir · Keyfiyyət skoru: 95%

## 📊 Nəticələr

Regional uyğunluq faizi:
- 🟢 Bakı: 74% 
- 🟡 Sumqayıt: 63%
- 🟡 Şirvan: 58%
- 🔴 Gəncə: 54%
- 🔴 Lənkəran: 49%

---
**Müəllif:** Shamil Remikhanov · Data Analyst  
**LinkedIn:** [linkedin.com/in/remikhanov-shamil](https://linkedin.com/in/remikhanov-shamil)
