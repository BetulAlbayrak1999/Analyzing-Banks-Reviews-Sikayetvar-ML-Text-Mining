"""
=============================================================
ADIM 5: TF-IDF + MAKİNE OGRENMESİ MODELLERİ
=============================================================
Bu script ne yapıyor?
TF-IDF Vektörizasyon:
temiz_metin sütununu sayısal matrise çevirir.
Her kelimeye bir ağırlık verir — sık geçen ama anlamsız 
kelimeler düşük, nadir ama ayırt edici kelimeler yüksek ağırlık alır.

Model, her banka için ayrı ayrı:
Logistic Regression: Hızlı, yorumlanabilir, baseline
Random Forest: Doğrusal olmayan ilişkileri yakalar
SVM (LinearSVC): Metin sınıflandırmada güçlü

Her model için üretilen metrikler:
Accuracy: Genel doğruluk
Precision: Çözüldü dediğinde ne kadar haklı
Recall: Gerçek çözülenler içinden ne kadarını buldu
F1 Score: Precision + Recall dengesi
ROC-AUC: Sınıfları ayırt etme gücü
CV F1 (5-fold): Modelin genellenebilirliği

H3 Hipotezi kontrolü: 
Script sonunda her modelin %70 Accuracy eşiğini geçip geçmediğini otomatik raporlar.


Bu script:
1. LDA ciktili veriyi yukler (veri_lda.csv)
2. TF-IDF ile metin temsili olusturur
3. Hedef degisken: is_resolved (Cozuldu / Bilinmiyor)
4. 3 model egitir: Logistic Regression, Random Forest, SVM
5. Her model icin: Accuracy, Precision, Recall, F1, ROC-AUC
6. Karisiklik matrisi ve ROC egrisi gorsellestirir
7. Ozellik onem analizi yapar
8. Tum modelleri kaydeder (.joblib)
9. Karsilastirmali sonuc tablosu uretir

CALISTIRMA: python adim5_tfidf_ml.py

GEREKLI:
  pip install scikit-learn imbalanced-learn joblib
=============================================================
"""

"""
=============================================================
DÜZELTMELER:
  1. tkinter / Tcl_AsyncDelete hatası:
     matplotlib.use("Agg") ile GUI backend tamamen devre dışı bırakıldı.
     Bu satır import'lardan ÖNCE ve EN BAŞA yazılmalıdır.
  2. Grafik yazıları artık doğru Türkçe karakterlerle yazılıyor.

ÇALIŞTIRMA: python adim5_tfidf_ml.py
=============================================================
"""

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# Türkçe karakter için font
plt.rcParams["font.family"]       = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd
import numpy as np
import os, warnings, json
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.ensemble                import RandomForestClassifier
from sklearn.svm                     import LinearSVC
from sklearn.model_selection         import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics                 import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import joblib

os.makedirs("models",          exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

# ----------------------------------------------------------
# PARAMETRELER
# ----------------------------------------------------------
TEST_BOYUTU        = 0.20
RANDOM_STATE       = 42
CV_FOLD            = 5
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM        = (1, 2)

BANKA_SIRASI = ["VakifBank", "IsBankasi", "KuveytTurk"]

# Grafiklerde gösterilecek Türkçe banka adları
BANKA_TR = {
    "VakifBank":  "VakıfBank",
    "IsBankasi":  "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}

RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBankasi":  "#B71C1C",
    "KuveytTurk": "#1B5E20",
}

MODELLER = {
    "Lojistik_Regresyon": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE,
        class_weight="balanced", C=1.0,
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1, max_depth=20,
    ),
    "SVM_Dogrusal": LinearSVC(
        random_state=RANDOM_STATE, class_weight="balanced",
        max_iter=2000, C=1.0,
    ),
}

# Grafiklerde kullanılacak model renkleri
MODEL_RENK = {
    "Lojistik_Regresyon": "#e41a1c",
    "Random_Forest":      "#377eb8",
    "SVM_Dogrusal":       "#4daf4a",
}

# ----------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ----------------------------------------------------------

def model_degerlendir(model, X_test, y_test, model_adi, banka):
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec     = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1      = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

    roc_auc  = None
    y_prob   = None
    try:
        if hasattr(model, "predict_proba"):
            y_prob  = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob  = model.decision_function(X_test)
        if y_prob is not None and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        pass

    sonuc = {
        "Banka":     banka,
        "Banka_TR":  BANKA_TR.get(banka, banka),
        "Model":     model_adi,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1_Skoru":  round(f1,   4),
        "ROC_AUC":   round(roc_auc, 4) if roc_auc else None,
    }

    print(f"    Doğruluk (Accuracy) : {acc:.4f}")
    print(f"    Kesinlik (Precision): {prec:.4f}")
    print(f"    Duyarlılık (Recall) : {rec:.4f}")
    print(f"    F1 Skoru            : {f1:.4f}")
    if roc_auc:
        print(f"    ROC-AUC             : {roc_auc:.4f}")

    return sonuc, y_pred, y_prob


def karisiklik_matrisi_ciz(y_test, y_pred, sinif_adlari,
                            model_adi, banka):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(sinif_adlari, fontsize=10)
    ax.set_yticklabels(sinif_adlari, fontsize=10)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    banka_tr = BANKA_TR.get(banka, banka)
    ax.set_title(f"{banka_tr} — {model_adi}\nKarışıklık Matrisi",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Gerçek Sınıf")
    ax.set_xlabel("Tahmin Edilen Sınıf")
    plt.tight_layout()
    fname = f"results/figures/karisiklik_{banka.lower()}_{model_adi.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close("all")   # ← tüm figürleri kapat, tkinter referansı bırakma


def ozellik_onem_ciz(model, model_adi, banka, vectorizer, topn=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    try:
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return

        idx     = np.argsort(importances)[-topn:]
        top_f   = feature_names[idx]
        top_imp = importances[idx]

        banka_tr = BANKA_TR.get(banka, banka)
        fig, ax  = plt.subplots(figsize=(8, 6))
        ax.barh(top_f, top_imp,
                color=RENKLER[banka], edgecolor="none")
        ax.set_title(f"{banka_tr} — {model_adi}\nEn Önemli {topn} TF-IDF Özelliği",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Önem Skoru (|Katsayı|)")
        plt.tight_layout()
        fname = f"results/figures/onem_{banka.lower()}_{model_adi.lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"      Özellik önem grafik hatası: {e}")


# ===========================================================
# ANA PROGRAM — Windows multiprocessing için zorunlu
# ===========================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. VERİ YÜKLE
    # --------------------------------------------------------
    print("Veri yükleniyor...")
    if os.path.exists("data/processed/veri_lda.csv"):
        veri = pd.read_csv("data/processed/veri_lda.csv", encoding="utf-8-sig")
        print("  veri_lda.csv yüklendi")
    else:
        veri = pd.read_csv("data/processed/veri_temiz.csv", encoding="utf-8-sig")
        print("  veri_temiz.csv yüklendi")

    print(f"  Toplam satır: {len(veri)}")

    # Hedef değişken
    veri = veri[veri["is_resolved"].notna()].copy()
    veri["hedef"] = (veri["is_resolved"].str.strip() == "Çözüldü").astype(int)

    print(f"\n  Hedef dağılımı:")
    print(f"  Çözüldü    (1): {(veri['hedef']==1).sum()}")
    print(f"  Bilinmiyor (0): {(veri['hedef']==0).sum()}")

    # Boş metin temizle
    veri = veri[
        veri["temiz_metin"].notna() &
        (veri["temiz_metin"].str.strip() != "")
    ].copy()
    print(f"  Geçerli satır: {len(veri)}")

    tum_sonuclar = []
    banka_roc    = {}

    # --------------------------------------------------------
    # 2. HER BANKA İÇİN MODEL EĞİTİMİ
    # --------------------------------------------------------
    for banka in BANKA_SIRASI:
        banka_tr = BANKA_TR.get(banka, banka)
        print(f"\n{'='*60}")
        print(f"  BANKA: {banka_tr}")
        print(f"{'='*60}")

        alt = veri[veri["banka_label"] == banka].copy()
        print(f"  Toplam şikayet: {len(alt)}")

        if len(alt) < 50:
            print("  UYARI: Yeterli veri yok, atlanıyor!")
            continue

        X = alt["temiz_metin"].values
        y = alt["hedef"].values

        coz_oran = (y == 1).sum() / len(y)
        print(f"  Çözülme oranı: %{coz_oran*100:.1f}")
        print(f"  Sınıf dağılımı → Bilinmiyor(0): {(y==0).sum()}  "
              f"Çözüldü(1): {(y==1).sum()}")

        # Train / Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_BOYUTU,
            random_state=RANDOM_STATE, stratify=y,
        )
        print(f"  Eğitim: {len(X_train)}  Test: {len(X_test)}")

        # TF-IDF
        print(f"\n  TF-IDF vektörizasyon...")
        tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM,
            sublinear_tf=True,
            min_df=2,
        )
        X_train_v = tfidf.fit_transform(X_train)
        X_test_v  = tfidf.transform(X_test)
        print(f"  TF-IDF matrisi boyutu: {X_train_v.shape}")

        joblib.dump(tfidf, f"models/tfidf_{banka.lower()}.joblib")

        banka_roc[banka] = {"model": [], "fpr": [], "tpr": [], "auc": []}
        sinif_adlari = ["Bilinmiyor", "Çözüldü"]

        # --------------------------------------------------------
        # 3. MODEL DÖNGÜSÜ
        # --------------------------------------------------------
        for model_adi, model in MODELLER.items():
            print(f"\n  ── {model_adi} ──")

            model.fit(X_train_v, y_train)

            sonuc, y_pred, y_prob = model_degerlendir(
                model, X_test_v, y_test, model_adi, banka
            )

            # Cross-validation
            cv = StratifiedKFold(n_splits=CV_FOLD, shuffle=True,
                                 random_state=RANDOM_STATE)
            cv_f1 = cross_val_score(
                model, X_train_v, y_train,
                cv=cv, scoring="f1_weighted", n_jobs=1
            )
            sonuc["CV_F1_Ort"] = round(cv_f1.mean(), 4)
            sonuc["CV_F1_Std"] = round(cv_f1.std(),  4)
            print(f"    CV F1 ({CV_FOLD}-kat): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

            # Sınıflandırma raporu
            rapor = classification_report(
                y_test, y_pred,
                target_names=sinif_adlari,
                zero_division=0
            )
            print(f"\n    Sınıflandırma Raporu:\n{rapor}")

            tum_sonuclar.append(sonuc)

            # Karışıklık matrisi
            karisiklik_matrisi_ciz(
                y_test, y_pred, sinif_adlari, model_adi, banka
            )

            # Özellik önem
            ozellik_onem_ciz(model, model_adi, banka, tfidf)

            # ROC verisi
            if y_prob is not None and len(np.unique(y_test)) == 2:
                try:
                    if hasattr(model, "predict_proba"):
                        yp = model.predict_proba(X_test_v)[:, 1]
                    else:
                        yp = model.decision_function(X_test_v)
                    fpr, tpr, _ = roc_curve(y_test, yp)
                    auc_val     = roc_auc_score(y_test, yp)
                    banka_roc[banka]["model"].append(model_adi)
                    banka_roc[banka]["fpr"].append(fpr.tolist())
                    banka_roc[banka]["tpr"].append(tpr.tolist())
                    banka_roc[banka]["auc"].append(round(auc_val, 4))
                except Exception:
                    pass

            # Model kaydet
            joblib.dump(model, f"models/{model_adi.lower()}_{banka.lower()}.joblib")
            print(f"    → Model kaydedildi: models/{model_adi.lower()}_{banka.lower()}.joblib")

        # --------------------------------------------------------
        # 4. ROC EĞRİSİ GRAFİĞİ
        # --------------------------------------------------------
        if banka_roc[banka]["model"]:
            fig, ax = plt.subplots(figsize=(7, 6))
            for i, madi in enumerate(banka_roc[banka]["model"]):
                fpr = banka_roc[banka]["fpr"][i]
                tpr = banka_roc[banka]["tpr"][i]
                auc = banka_roc[banka]["auc"][i]
                ax.plot(fpr, tpr, linewidth=2,
                        color=MODEL_RENK.get(madi, "gray"),
                        label=f"{madi}  (AUC = {auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
                    label="Rastgele Sınıflayıcı")
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=11)
            ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=11)
            ax.set_title(f"{banka_tr}\nROC Eğrisi Karşılaştırması",
                         fontsize=12, fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            plt.tight_layout()
            plt.savefig(f"results/figures/roc_{banka.lower()}.png",
                        dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"\n  → ROC grafiği kaydedildi: results/figures/roc_{banka.lower()}.png")

    # --------------------------------------------------------
    # 5. KARŞILAŞTIRMA GRAFİKLERİ
    # --------------------------------------------------------
    if tum_sonuclar:
        sonuc_df = pd.DataFrame(tum_sonuclar)
        sonuc_df.to_csv("results/reports/model_sonuclari.csv",
                        index=False, encoding="utf-8-sig")
        print("\n→ results/reports/model_sonuclari.csv kaydedildi")

        bankalar_tr  = [BANKA_TR[b] for b in BANKA_SIRASI if b in sonuc_df["Banka"].values]
        modeller_lst = list(MODELLER.keys())
        x            = np.arange(len(bankalar_tr))
        genislik     = 0.25
        ofsetler     = np.linspace(-genislik, genislik, len(modeller_lst))

        for metrik, baslik, ylabel in [
            ("F1_Skoru",  "F1 Skoru Karşılaştırması",  "F1 Skoru (Ağırlıklı Ortalama)"),
            ("Accuracy",  "Doğruluk (Accuracy) Karşılaştırması", "Doğruluk (Accuracy)"),
        ]:
            fig, ax = plt.subplots(figsize=(12, 5))
            for i, model_adi in enumerate(modeller_lst):
                degerler = []
                for banka in BANKA_SIRASI:
                    if banka not in sonuc_df["Banka"].values:
                        continue
                    alt = sonuc_df[(sonuc_df["Banka"] == banka) &
                                   (sonuc_df["Model"] == model_adi)]
                    degerler.append(alt[metrik].values[0] if len(alt) > 0 else 0)

                bars = ax.bar(
                    x + ofsetler[i], degerler, genislik,
                    label=model_adi,
                    color=MODEL_RENK.get(model_adi, "gray"),
                    edgecolor="white",
                )
                ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

            ax.set_xticks(x)
            ax.set_xticklabels(bankalar_tr, fontsize=12)
            ax.set_ylim([0, 1.12])
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"Model Karşılaştırması — {baslik}",
                         fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.axhline(0.70, color="gray", linestyle="--",
                       linewidth=1.2, alpha=0.7, label="H3 Eşiği: 0.70")
            plt.tight_layout()
            fname = f"results/figures/karsilastirma_{metrik.lower()}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"→ {fname} kaydedildi")

        # --------------------------------------------------------
        # 6. TERMINAL ÖZET
        # --------------------------------------------------------
        print("\n" + "─"*70)
        print("SONUÇ TABLOSU")
        print("─"*70)
        goster = sonuc_df[["Banka_TR","Model","Accuracy","F1_Skoru",
                            "ROC_AUC","CV_F1_Ort","CV_F1_Std"]]
        print(goster.to_string(index=False))

        print("\n" + "─"*70)
        print("H3 HİPOTEZİ: Modeller ≥ %70 Accuracy ile tahmin edebilmeli")
        print("─"*70)
        for _, row in sonuc_df.iterrows():
            acc    = row["Accuracy"]
            durum  = "✓ SAĞLANDI" if acc >= 0.70 else "✗ SAĞLANAMADI"
            print(f"  {row['Banka_TR']:14s} | {row['Model']:22s} | "
                  f"Acc={acc:.4f}  {durum}")

    # --------------------------------------------------------
    # ÖZET
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("ADIM 5 TAMAMLANDI!")
    print()
    print("Üretilen dosyalar:")
    print("  models/tfidf_*.joblib")
    print("  models/<model>_<banka>.joblib")
    print("  results/reports/model_sonuclari.csv  ← tez ana tablosu")
    print("  results/figures/karisiklik_*.png     ← karışıklık matrisleri")
    print("  results/figures/roc_*.png            ← ROC eğrileri")
    print("  results/figures/onem_*.png           ← özellik önem grafikleri")
    print("  results/figures/karsilastirma_*.png  ← özet karşılaştırma")
    print()
    print("Sıradaki adım: python adim6_karsilastirma.py")
    print("="*60)