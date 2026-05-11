"""
=============================================================
ADIM 5 (UZMAN): TF-IDF + MAKİNE ÖĞRENMESİ MODELLERİ
=============================================================
Bankacılık Sektörü Müşteri Şikayet Analizi — 2025
Kuveyt Türk | VakıfBank | İşBankası

TEMEL TASARIM KARARLARI:
  ✓ TF-IDF (10.000 özellik, unigram+bigram, sublinear_tf)
  ✓ Meta özellikler (TF-IDF matrisiyle sparse hstack):
      - token_uzunluk : metin uzunluğu (çözülenler daha uzun)
      - view_log      : log(görüntülenme+1)
      - has_reply     : şirket yanıtladı mı? (korelasyon: 0.31)
      NOT: satisfaction kullanılmadı (hedef ile leakage riski)
  ✓ 3 algoritma — her banka ayrı:
      - Lojistik Regresyon  (LR)  : baseline, yorumlanabilir
      - Random Forest       (RF)  : doğrusal olmayan ilişkiler
      - SVM (LinearSVC)    (SVM) : metin sınıflandırmada güçlü
  ✓ class_weight='balanced'  → sınıf dengesizliği (1:1.9) çözümü
  ✓ CalibratedClassifierCV   → SVM için olasılık + ROC-AUC
  ✓ Metrikler: Accuracy, Precision, Recall, F1, ROC-AUC, CV-F1(5-kat)
  ✓ H3 hipotezi: ≥%70 Accuracy eşiği kontrolü
  ✓ Özellik önemi: LR/SVM katsayıları, RF Gini skoru
  ✓ SVM katsayısı: calibrated_classifiers_ üzerinden ortalama

ÜRETİLEN ÇIKTILAR:
  models/tfidf_{banka}.joblib
  models/{model}_{banka}.joblib
  results/reports/model_sonuclari.csv
  results/reports/ozellik_onem.csv
  results/figures/23_roc_egrisi.png         ← 3×4 grid, her algoritma ayrı
  results/figures/24_karisiklik_matrisi.png ← 3×3 grid, köşegen altın çerçeve
  results/figures/25_model_karsilastirma.png← 6 metrik, ★ en iyi algoritma
  results/figures/26_ozellik_onem.png       ← 3×3 grid, LR/SVM/RF ayrı
  results/figures/27_h3_hipotez_panel.png   ← Banka×algoritma H3 tablosu
  results/figures/28_cv_performans.png      ← 3×3 boxplot+jitter grid
  results/figures/29_radar_algoritma.png    ← Koyu tema radar grafiği

ÇALIŞTIRMA: python adim5_tfidf.py
=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 150

import pandas as pd
import numpy as np
import os, warnings, json

warnings.filterwarnings("ignore")

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
import joblib

os.makedirs("models",          exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

# ============================================================
# SABİTLER
# ============================================================
BANKA_SIRASI = ["VakifBank", "IsBank", "KuveytTurk"]
BANKA_TR = {
    "VakifBank":  "VakıfBank",
    "IsBank":     "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}
BANKA_RENK = {
    "VakifBank":  "#1565C0",
    "IsBank":     "#C62828",
    "KuveytTurk": "#2E7D32",
}

TEST_BOYUTU  = 0.20
RANDOM_STATE = 42
CV_FOLD      = 5
TFIDF_FEATS  = 10_000
H3_ESIK      = 0.70

MODEL_ISIM = {
    "Lojistik_Regresyon": "Lojistik Regresyon",
    "Random_Forest":      "Random Forest",
    "SVM_Dogrusal":       "SVM (Doğrusal Çekirdek)",
}
MODEL_RENK = {
    "Lojistik_Regresyon": "#E53935",
    "Random_Forest":      "#1E88E5",
    "SVM_Dogrusal":       "#43A047",
}
MODEL_MARKER = {
    "Lojistik_Regresyon": "o",
    "Random_Forest":      "s",
    "SVM_Dogrusal":       "^",
}
MODEL_LISTESI = ["Lojistik_Regresyon", "Random_Forest", "SVM_Dogrusal"]
SINIF_ADLARI  = ["Çözülmedi", "Çözüldü"]

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def meta_ozellik_hazirla(df):
    """TF-IDF'e eklenecek sayısal meta özellikler."""
    df = df.copy()
    df["token_uzunluk"] = df["temiz_metin"].fillna("").apply(
        lambda x: len(x.split()))
    df["view_log"] = np.log1p(
        pd.to_numeric(df["view_count"], errors="coerce").fillna(0))
    df["has_reply"] = df["company_reply"].notna().astype(float)
    return df[["token_uzunluk", "view_log", "has_reply"]].fillna(0).values


def model_degerlendir(model, X_te, y_te, model_adi, banka):
    """Model metrikleri hesapla."""
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    prec   = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    f1_coz = f1_score(y_te, y_pred, pos_label=1, average="binary", zero_division=0)

    auc, y_prob = None, None
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_te)
        if y_prob is not None and len(np.unique(y_te)) == 2:
            auc = roc_auc_score(y_te, y_prob)
    except Exception:
        pass

    return {
        "Banka":      banka,
        "Banka_TR":   BANKA_TR[banka],
        "Model":      model_adi,
        "Accuracy":   round(acc, 4),
        "Precision":  round(prec, 4),
        "Recall":     round(rec, 4),
        "F1_Agirlik": round(f1, 4),
        "F1_Cozuldu": round(f1_coz, 4),
        "ROC_AUC":    round(auc, 4) if auc else None,
        "H3_Sagland": acc >= H3_ESIK,
    }, y_pred, y_prob


def svm_katsayi_al(model, feat_names, topn=15):
    """CalibratedClassifierCV → ortalama katsayı."""
    coefs = [cc.estimator.coef_[0]
             for cc in model.calibrated_classifiers_
             if hasattr(cc.estimator, "coef_")]
    if not coefs:
        return {}
    coef     = np.mean(coefs, axis=0)
    pos_idx  = np.argsort(coef)[-topn:][::-1]
    neg_idx  = np.argsort(coef)[:topn]
    return {
        "pos": [(feat_names[i], float(coef[i])) for i in pos_idx],
        "neg": [(feat_names[i], float(coef[i])) for i in neg_idx],
        "tip": "lr_svm",
    }


def onem_hesapla(model, model_adi, tfidf, topn=15):
    """LR / RF / SVM için özellik önem sözlüğü döndür."""
    feat_names = np.array(
        list(tfidf.get_feature_names_out()) +
        ["token_uzunluk", "view_log", "has_reply"]
    )
    try:
        if model_adi == "SVM_Dogrusal":
            return svm_katsayi_al(model, feat_names, topn)
        if hasattr(model, "coef_"):          # LR
            coef    = model.coef_[0]
            pos_idx = np.argsort(coef)[-topn:][::-1]
            neg_idx = np.argsort(coef)[:topn]
            return {
                "pos": [(feat_names[i], float(coef[i])) for i in pos_idx],
                "neg": [(feat_names[i], float(coef[i])) for i in neg_idx],
                "tip": "lr_svm",
            }
        if hasattr(model, "feature_importances_"):  # RF
            imp = model.feature_importances_
            idx = np.argsort(imp)[-topn:][::-1]
            return {
                "onem": [(feat_names[i], float(imp[i])) for i in idx],
                "tip": "rf",
            }
    except Exception:
        pass
    return {}


# ============================================================
# ANA PROGRAM
# ============================================================
if __name__ == "__main__":

    print("=" * 65)
    print("ADIM 5 (UZMAN): TF-IDF + MAKİNE ÖĞRENMESİ")
    print("=" * 65)

    # ----------------------------------------------------------
    # 1. VERİ YÜKLE
    # ----------------------------------------------------------
    print("\n[Veri yükleniyor...]")
    kaynak = ("data/processed/veri_lda.csv"
              if os.path.exists("data/processed/veri_lda.csv")
              else "data/processed/veri_temiz.csv")
    veri = pd.read_csv(kaynak, encoding="utf-8-sig")
    veri = veri[veri["is_resolved"].notna()].copy()
    veri["hedef"] = (veri["is_resolved"].str.strip() == "Çözüldü").astype(int)

    print(f"  Toplam : {len(veri):,} şikayet")
    print(f"  Çözüldü    (1): {veri['hedef'].sum():,}")
    print(f"  Çözülmedi  (0): {(veri['hedef']==0).sum():,}")
    print(f"  Dengesizlik: 1:{(veri['hedef']==0).sum()/veri['hedef'].sum():.2f}")

    # ----------------------------------------------------------
    # 2. MODEL TANIMLARI
    # ----------------------------------------------------------
    def modeller_olustur():
        return {
            "Lojistik_Regresyon": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE,
                class_weight="balanced", C=0.5, solver="saga",
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1,
                max_depth=15, min_samples_leaf=3,
            ),
            "SVM_Dogrusal": CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_STATE,
                          class_weight="balanced",
                          max_iter=5000, C=0.5),
                cv=3,
            ),
        }

    # ----------------------------------------------------------
    # 3. HER BANKA İÇİN EĞİTİM
    # ----------------------------------------------------------
    tum_sonuclar = []
    roc_verileri = {}   # {banka: [{model,fpr,tpr,auc},...]}
    cm_verileri  = {}   # {banka: {model: cm_array}}
    cv_verileri  = {}   # {banka: {model: cv_f1_array}}
    onem_verileri = {}  # {banka: {model: onem_dict}}

    for banka in BANKA_SIRASI:
        bl = BANKA_TR[banka]
        print(f"\n{'='*60}\n  {bl}\n{'='*60}")

        alt = veri[veri["banka_label"] == bl].copy()
        alt = alt[alt["temiz_metin"].notna() &
                  (alt["temiz_metin"].str.strip() != "")].copy()

        print(f"  Şikayet: {len(alt):,}  "
              f"Çözüldü: {alt['hedef'].sum():,} "
              f"(%{alt['hedef'].mean()*100:.1f})")

        if len(alt) < 50:
            print("  ⚠ Yetersiz veri, atlanıyor!")
            continue

        X_text = alt["temiz_metin"].values
        X_meta = meta_ozellik_hazirla(alt)
        y      = alt["hedef"].values

        (X_tr_t, X_te_t,
         X_tr_m, X_te_m,
         y_tr, y_te) = train_test_split(
            X_text, X_meta, y,
            test_size=TEST_BOYUTU,
            random_state=RANDOM_STATE, stratify=y,
        )
        print(f"  Eğitim: {len(X_tr_t):,}  Test: {len(X_te_t):,}")

        # TF-IDF
        print("  TF-IDF vektörizasyon...")
        tfidf = TfidfVectorizer(
            max_features=TFIDF_FEATS, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2,
        )
        X_tr_v = hstack([tfidf.fit_transform(X_tr_t), csr_matrix(X_tr_m)])
        X_te_v = hstack([tfidf.transform(X_te_t),     csr_matrix(X_te_m)])
        X_all_v = hstack([tfidf.transform(X_text),    csr_matrix(X_meta)])
        print(f"  Özellik boyutu: {X_tr_v.shape[1]:,} "
              f"(TF-IDF:{TFIDF_FEATS:,} + meta:3)")

        joblib.dump(tfidf, f"models/tfidf_{banka.lower()}.joblib")

        roc_verileri[banka]  = []
        cm_verileri[banka]   = {}
        cv_verileri[banka]   = {}
        onem_verileri[banka] = {}

        cv = StratifiedKFold(n_splits=CV_FOLD, shuffle=True,
                             random_state=RANDOM_STATE)
        modeller = modeller_olustur()

        for model_adi, model in modeller.items():
            print(f"\n  ── {MODEL_ISIM[model_adi]} ──")
            model.fit(X_tr_v, y_tr)

            sonuc, y_pred, y_prob = model_degerlendir(
                model, X_te_v, y_te, model_adi, banka)

            print(f"    Accuracy : {sonuc['Accuracy']:.4f}  "
                  f"{'✓ H3' if sonuc['H3_Sagland'] else '✗ H3'}")
            print(f"    F1 (ağır): {sonuc['F1_Agirlik']:.4f}")
            if sonuc["ROC_AUC"]:
                print(f"    ROC-AUC  : {sonuc['ROC_AUC']:.4f}")

            # Cross-validation
            cv_f1 = cross_val_score(model, X_all_v, y,
                                    cv=cv, scoring="f1_weighted", n_jobs=1)
            sonuc["CV_F1_Ort"] = round(cv_f1.mean(), 4)
            sonuc["CV_F1_Std"] = round(cv_f1.std(), 4)
            print(f"    CV-F1 ({CV_FOLD}-kat): "
                  f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

            rapor = classification_report(
                y_te, y_pred, target_names=SINIF_ADLARI, zero_division=0)
            print(f"\n  Sınıflandırma Raporu:\n{rapor}")

            tum_sonuclar.append(sonuc)

            # ROC
            if y_prob is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_te, y_prob)
                    roc_verileri[banka].append({
                        "model": model_adi,
                        "fpr":   fpr, "tpr": tpr,
                        "auc":   sonuc["ROC_AUC"],
                    })
                except Exception:
                    pass

            cm_verileri[banka][model_adi]   = confusion_matrix(y_te, y_pred)
            cv_verileri[banka][model_adi]   = cv_f1
            onem_verileri[banka][model_adi] = onem_hesapla(model, model_adi, tfidf)

            joblib.dump(model,
                        f"models/{model_adi.lower()}_{banka.lower()}.joblib")

    # Sonuç tablosu kaydet
    sonuc_df = pd.DataFrame(tum_sonuclar)
    sonuc_df.to_csv("results/reports/model_sonuclari.csv",
                    index=False, encoding="utf-8-sig")

    # Özellik önem raporu kaydet
    onem_satirlari = []
    for banka, m_dict in onem_verileri.items():
        for model_adi, onem in m_dict.items():
            if not onem:
                continue
            if "pos" in onem:
                for t, k in onem["pos"][:10]:
                    onem_satirlari.append({"Banka": BANKA_TR[banka],
                                           "Model": model_adi,
                                           "Yon": "Çözüldü",
                                           "Terim": str(t),
                                           "Agirlik": round(k, 5)})
                for t, k in onem["neg"][:10]:
                    onem_satirlari.append({"Banka": BANKA_TR[banka],
                                           "Model": model_adi,
                                           "Yon": "Çözülmedi",
                                           "Terim": str(t),
                                           "Agirlik": round(k, 5)})
    pd.DataFrame(onem_satirlari).to_csv(
        "results/reports/ozellik_onem.csv",
        index=False, encoding="utf-8-sig")

    print("\n✓ model_sonuclari.csv  ✓ ozellik_onem.csv kaydedildi")

    # ============================================================
    # 4. FİGÜRLER
    # ============================================================
    print("\n[Figürler oluşturuluyor...]")

    # ─── FİGÜR 23: ROC Eğrisi — 3×4 Grid ─────────────────────
    # Satır=Banka  Sütun0-2=Her algoritma ayrı  Sütun3=3 algo birlikte
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "ROC Eğrisi Analizi: Her Algoritma Ayrı\n"
        "İlk 3 sütun: Her banka-algoritma çifti  |  Son sütun: 3 algoritmanın banka içi karşılaştırması\n"
        "AUC=1.0 mükemmel · AUC=0.5 rastgele tahmin · Optimal nokta: Youden J istatistiği",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35)

    for ri, banka in enumerate(BANKA_SIRASI):
        bl    = BANKA_TR[banka]
        brenk = BANKA_RENK[banka]

        for ci, model_adi in enumerate(MODEL_LISTESI):
            ax = fig.add_subplot(gs[ri, ci])
            ax.set_facecolor("white")

            roc_lst = [r for r in roc_verileri.get(banka, [])
                       if r["model"] == model_adi]
            if not roc_lst:
                ax.text(0.5, 0.5, "Veri yok", ha="center",
                        transform=ax.transAxes)
                continue
            roc = roc_lst[0]
            fpr = np.array(roc["fpr"])
            tpr = np.array(roc["tpr"])
            auc = roc["auc"]

            ax.fill_between(fpr, tpr, alpha=0.12,
                            color=MODEL_RENK[model_adi])
            ax.plot(fpr, tpr, lw=2.8, color=MODEL_RENK[model_adi])
            ax.plot([0,1],[0,1],"--", color="#BBBBBB", lw=1.2)

            # Optimal nokta (Youden J)
            j_idx = np.argmax(tpr - fpr)
            ax.scatter(fpr[j_idx], tpr[j_idx], s=100,
                       color=MODEL_RENK[model_adi],
                       zorder=6, edgecolors="white", lw=2)
            ax.annotate(
                f"Optimal\n({fpr[j_idx]:.2f},{tpr[j_idx]:.2f})",
                xy=(fpr[j_idx], tpr[j_idx]),
                xytext=(fpr[j_idx]+0.07, tpr[j_idx]-0.12),
                fontsize=7.5, color=MODEL_RENK[model_adi],
                arrowprops=dict(arrowstyle="->",
                                color=MODEL_RENK[model_adi], lw=1),
            )

            ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
            ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=8)
            ax.set_ylabel("Doğru Pozitif Oranı (TPR)"
                          if ci == 0 else "", fontsize=8)
            ax.set_title(f"{bl}\n{MODEL_ISIM[model_adi]}",
                         fontsize=9.5, fontweight="bold",
                         color=MODEL_RENK[model_adi])
            ax.text(0.04, 0.97, f"AUC = {auc:.3f}",
                    transform=ax.transAxes, va="top",
                    fontsize=10, fontweight="bold",
                    color=MODEL_RENK[model_adi],
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.9,
                              edgecolor=MODEL_RENK[model_adi], lw=1.5))
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(alpha=0.2, ls="--")

        # Son sütun: 3 algoritma birlikte
        ax_all = fig.add_subplot(gs[ri, 3])
        ax_all.set_facecolor("white")
        for model_adi in MODEL_LISTESI:
            roc_lst = [r for r in roc_verileri.get(banka, [])
                       if r["model"] == model_adi]
            if not roc_lst:
                continue
            roc = roc_lst[0]
            ax_all.plot(np.array(roc["fpr"]), np.array(roc["tpr"]),
                        lw=2.2, color=MODEL_RENK[model_adi],
                        marker=MODEL_MARKER[model_adi],
                        markevery=20, ms=5,
                        label=f"{MODEL_ISIM[model_adi][:8]}..  "
                              f"AUC={roc['auc']:.3f}")
        ax_all.plot([0,1],[0,1],"--",color="#BBBBBB",lw=1)
        ax_all.set_xlim([0,1]); ax_all.set_ylim([0,1.02])
        ax_all.set_xlabel("FPR", fontsize=8)
        ax_all.set_title(f"{bl}\n3 Algoritma Karşılaştırması",
                         fontsize=9.5, fontweight="bold", color=brenk)
        ax_all.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
        ax_all.spines[["top","right"]].set_visible(False)
        ax_all.grid(alpha=0.2, ls="--")

    plt.savefig("results/figures/23_roc_egrisi.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 23_roc_egrisi.png")

    # ─── FİGÜR 24: Karışıklık Matrisleri — 3×3 Grid ──────────
    fig, axes = plt.subplots(3, 3, figsize=(17, 16))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "Karışıklık Matrisleri: Banka × Algoritma\n"
        "Satır: Gerçek sınıf  |  Sütun: Tahmin edilen sınıf  |  "
        "Altın çerçeve: Doğru tahminler (köşegen)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    # Sütun başlıkları
    for ci, model_adi in enumerate(MODEL_LISTESI):
        fig.text(0.18 + ci*0.27, 1.005,
                 MODEL_ISIM[model_adi], ha="center",
                 fontsize=13, fontweight="bold",
                 color=MODEL_RENK[model_adi],
                 transform=fig.transFigure)

    for ri, banka in enumerate(BANKA_SIRASI):
        bl    = BANKA_TR[banka]
        brenk = BANKA_RENK[banka]
        cmap_name = {"VakifBank":"Blues","IsBank":"Reds",
                     "KuveytTurk":"Greens"}[banka]

        for ci, model_adi in enumerate(MODEL_LISTESI):
            ax = axes[ri, ci]
            ax.set_facecolor("white")
            cm = cm_verileri.get(banka, {}).get(model_adi)
            if cm is None:
                ax.axis("off"); continue

            total  = cm.sum()
            thresh = cm.max() / 2.0
            ax.imshow(cm, cmap=cmap_name, interpolation="nearest",
                      vmin=0, vmax=cm.max())

            for i in range(2):
                for j in range(2):
                    pct = cm[i,j] / total * 100
                    ax.text(j, i,
                            f"{cm[i,j]:,}\n({pct:.1f}%)",
                            ha="center", va="center",
                            fontsize=12, fontweight="bold",
                            color="white" if cm[i,j]>thresh else "black")

            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            ax.set_xticklabels(SINIF_ADLARI, fontsize=9.5,
                               fontweight="bold")
            ax.set_yticklabels(SINIF_ADLARI, fontsize=9.5,
                               fontweight="bold")
            ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=9)
            ax.set_ylabel("Gerçek Sınıf" if ci==0 else "", fontsize=9)

            acc_row = sonuc_df[(sonuc_df["Banka"]==banka) &
                               (sonuc_df["Model"]==model_adi)]
            acc_val = acc_row["Accuracy"].values[0] if len(acc_row)>0 else 0

            ax.set_title(
                f"{bl}  ·  {MODEL_ISIM[model_adi]}\n"
                f"Accuracy = {acc_val:.3f}  "
                f"{'✓ H≥70%' if acc_val>=H3_ESIK else '✗ H<70%'}",
                fontsize=9.5, fontweight="bold",
                color=MODEL_RENK[model_adi],
            )
            # Köşegen altın çerçeve
            for i in range(2):
                ax.add_patch(plt.Rectangle(
                    (i-0.5, i-0.5), 1, 1, fill=False,
                    edgecolor="gold", lw=2.5, zorder=5))

    plt.tight_layout(h_pad=3.5, w_pad=2)
    plt.savefig("results/figures/24_karisiklik_matrisi.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 24_karisiklik_matrisi.png")

    # ─── FİGÜR 25: 6 Metrik Karşılaştırma — 2×3 Grid ─────────
    METRIKLER_6 = [
        ("Accuracy",   "Doğruluk (Accuracy)",    "Genel Doğruluk Oranı"),
        ("F1_Agirlik", "F1 Skoru (Ağırlıklı)",   "Ağırlıklı F1 Skoru"),
        ("ROC_AUC",    "ROC-AUC",                 "ROC Eğrisi Altı Alan"),
        ("CV_F1_Ort",  "CV F1 (5-kat ort.)",      "Çapraz Doğrulama F1"),
        ("Precision",  "Kesinlik (Precision)",    "Pozitif Tahmin Doğruluğu"),
        ("Recall",     "Duyarlılık (Recall)",     "Gerçek Pozitifleri Bulma"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "Model Performans Karşılaştırması: 6 Metrik × 3 Algoritma × 3 Banka\n"
        "Her renk bir algoritmayı temsil eder  ·  Kesik çizgi: Hipotez %70 eşiği  ·  "
        "★ = Her banka için en iyi algoritma",
        fontsize=13, fontweight="bold", y=1.01,
    )

    banka_labels = [BANKA_TR[b] for b in BANKA_SIRASI]
    x       = np.arange(len(banka_labels))
    genislik = 0.24
    ofsetler = [-genislik, 0, genislik]

    for (metrik, baslik, ylabel), ax in zip(METRIKLER_6, axes.flatten()):
        ax.set_facecolor("white")
        if metrik not in sonuc_df.columns:
            ax.text(0.5, 0.5, "Veri yok", ha="center",
                    transform=ax.transAxes)
            continue

        for i, model_adi in enumerate(MODEL_LISTESI):
            degerler, hatalar = [], []
            for banka in BANKA_SIRASI:
                alt = sonuc_df[(sonuc_df["Banka"]==banka) &
                               (sonuc_df["Model"]==model_adi)]
                val = float(alt[metrik].values[0]) \
                    if len(alt)>0 and pd.notna(alt[metrik].values[0]) else 0
                degerler.append(val)
                hatalar.append(
                    float(alt["CV_F1_Std"].values[0])
                    if metrik == "CV_F1_Ort" and len(alt)>0 else 0)

            bars = ax.bar(x + ofsetler[i], degerler,
                          genislik - 0.02,
                          label=MODEL_ISIM[model_adi],
                          color=MODEL_RENK[model_adi],
                          edgecolor="white", lw=1.5, alpha=0.88)
            if any(e > 0 for e in hatalar):
                ax.errorbar(x + ofsetler[i], degerler, yerr=hatalar,
                            fmt="none", color="#333",
                            capsize=4, lw=1.5)
            for bar, val in zip(bars, degerler):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.006,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold",
                            color=MODEL_RENK[model_adi])

        ax.set_xticks(x)
        ax.set_xticklabels(banka_labels, fontsize=11, fontweight="bold")
        ax.set_ylim([0.35, 1.10])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(baslik, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, ls="--")

        if metrik == "Accuracy":
            ax.axhline(H3_ESIK, color="#B71C1C",
                       ls="--", lw=2, alpha=0.85, zorder=0)
            ax.text(x[-1]+0.42, H3_ESIK+0.01,
                    "H\n%70", fontsize=8.5, color="#B71C1C",
                    fontweight="bold")

        # En iyi algoritma ★
        for bi, banka in enumerate(BANKA_SIRASI):
            best_val, best_mi = 0, 0
            for mi, model_adi in enumerate(MODEL_LISTESI):
                alt = sonuc_df[(sonuc_df["Banka"]==banka) &
                               (sonuc_df["Model"]==model_adi)]
                if len(alt)>0 and pd.notna(alt[metrik].values[0]):
                    val = float(alt[metrik].values[0])
                    if val > best_val:
                        best_val, best_mi = val, mi
            ax.text(bi + ofsetler[best_mi], 0.36, "★",
                    ha="center", fontsize=12,
                    color=MODEL_RENK[MODEL_LISTESI[best_mi]])

    fig.legend(
        handles=[Patch(color=MODEL_RENK[m], label=MODEL_ISIM[m])
                 for m in MODEL_LISTESI],
        loc="lower center", ncol=3, fontsize=11,
        framealpha=0.95, bbox_to_anchor=(0.5, -0.03),
        edgecolor="#CCC")

    plt.tight_layout(h_pad=4, w_pad=3)
    plt.savefig("results/figures/25_model_karsilastirma.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 25_model_karsilastirma.png")

    # ─── FİGÜR 26: Özellik Önemi — 3×3 Grid ──────────────────
    """
    fig, axes = plt.subplots(3, 3, figsize=(24, 22))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "Şekil 26 — Özellik Önemi Analizi: Her Algoritma Ayrı\n"
        "Lojistik Regresyon & SVM: katsayı büyüklüğü  |  "
        "Random Forest: Gini önem skoru\n"
        "Yeşil ↑ Çözüldü yönünde  ·  Kırmızı ↓ Çözülmedi yönünde  ·  "
        "🟠 meta özellik",
        fontsize=13, fontweight="bold", y=1.01,
    )
    for ci, model_adi in enumerate(MODEL_LISTESI):
        fig.text(0.15 + ci * 0.285, 1.005,
                 MODEL_ISIM[model_adi], ha="center",
                 fontsize=14, fontweight="bold",
                 color=MODEL_RENK[model_adi],
                 transform=fig.transFigure)

    META_OZELLIKLERI = {"has_reply", "token_uzunluk", "view_log"}

    for ri, banka in enumerate(BANKA_SIRASI):
        bl     = BANKA_TR[banka]
        brenk  = BANKA_RENK[banka]

        for ci, model_adi in enumerate(MODEL_LISTESI):
            ax     = axes[ri, ci]
            ax.set_facecolor("white")
            onem   = onem_verileri.get(banka, {}).get(model_adi, {})
            m_renk = MODEL_RENK[model_adi]

            if not onem:
                ax.text(0.5, 0.5, "Önem verisi yok",
                        ha="center", transform=ax.transAxes)
                ax.set_title(f"{bl}\n{MODEL_ISIM[model_adi]}",
                             fontsize=9, color=m_renk, fontweight="bold")
                continue

            if onem.get("tip") == "rf":
                items     = onem["onem"][:12]
                kelimeler = [str(x[0]) for x in items]
                skorlar   = [float(x[1]) for x in items]

                bar_renkler = ["#FF6B35" if k in META_OZELLIKLERI
                               else m_renk for k in kelimeler]
                ax.barh(range(len(kelimeler)), skorlar,
                        color=bar_renkler, alpha=0.82,
                        edgecolor="none", height=0.65)
                ax.set_yticks(range(len(kelimeler)))
                ax.set_yticklabels(kelimeler, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("Gini Önem Skoru", fontsize=8.5)
                ax.set_title(f"{bl} · {MODEL_ISIM[model_adi]}\n"
                             f"En Önemli {len(kelimeler)} Özellik",
                             fontsize=9.5, fontweight="bold",
                             color=m_renk)
                for i, val in enumerate(skorlar):
                    ax.text(val + max(skorlar)*0.01, i,
                            f"{val:.4f}", va="center", fontsize=7.5)
            else:
                # LR / SVM: iki yönlü
                pos_items = onem.get("pos", [])[:10]
                neg_items = onem.get("neg", [])[:10]
                pos_kw = [str(x[0]) for x in pos_items]
                pos_sk = [float(x[1]) for x in pos_items]
                neg_kw = [str(x[0]) for x in neg_items]
                neg_sk = [abs(float(x[1])) for x in neg_items]

                n_pos = len(pos_kw)

                pos_renkler = ["#FF6B35" if k in META_OZELLIKLERI
                               else "#43A047" for k in pos_kw]
                ax.barh(range(n_pos), pos_sk,
                        color=pos_renkler, alpha=0.82,
                        edgecolor="none", height=0.65)
                for i, val in enumerate(pos_sk):
                    ax.text(val + max(pos_sk+[0.01])*0.02, i,
                            f"{val:.3f}", va="center", fontsize=7.5,
                            color="#1B5E20")

                ax.axhline(n_pos - 0.5, color="#888",
                           lw=0.8, ls="--")
                ax.text(max(pos_sk+[0.01])*0.5, n_pos - 0.2,
                        "── Çözüldü ↑   ── Çözülmedi ↓ ──",
                        ha="center", fontsize=7.5, color="#888", style="italic")

                y_off = n_pos + 1
                neg_renkler = ["#FF6B35" if k in META_OZELLIKLERI
                               else "#E53935" for k in neg_kw]
                ax.barh(range(y_off, y_off + len(neg_kw)), neg_sk,
                        color=neg_renkler, alpha=0.82,
                        edgecolor="none", height=0.65)
                for i, val in enumerate(neg_sk):
                    ax.text(val + max(neg_sk+[0.01])*0.02,
                            y_off + i,
                            f"{val:.3f}", va="center", fontsize=7.5,
                            color="#B71C1C")

                tum_kw = pos_kw + [""] + neg_kw
                ax.set_yticks(range(len(tum_kw)))
                ax.set_yticklabels(tum_kw, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("|Katsayı| (model ağırlığı)", fontsize=8.5)
                ax.set_title(
                    f"{bl} · {MODEL_ISIM[model_adi]}\n"
                    "Yeşil: Çözüldü öngörücü  ·  Kırmızı: Çözülmedi öngörücü",
                    fontsize=9.5, fontweight="bold", color=m_renk)

            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.text(0.98, 0.01,
                    "🟠 = meta özellik",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, color="#FF6B35", style="italic")

    plt.tight_layout(h_pad=4, w_pad=3)
    plt.savefig("results/figures/26_ozellik_onem.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 26_ozellik_onem.png")

    # ─── FİGÜR 27: H3 Hipotezi Özet — Banka × Algoritma ──────
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "Şekil 27 — H3 Hipotezi: 'ML Modelleri ≥%70 Accuracy ile Çözüm Tahmini Yapabilmeli'\n"
        "Her banka için 3 algoritma ayrı gösterilmiştir  ·  Kesik çizgi: %70 eşiği  ·  "
        "✓ Sağlandı  ✗ Sağlanamadı",
        fontsize=13, fontweight="bold", y=1.01,
    )
    METRIKLER_H3 = [
        ("Accuracy",   "Doğruluk"),
        ("F1_Agirlik", "F1 Skoru"),
        ("ROC_AUC",    "ROC-AUC"),
        ("CV_F1_Ort",  "CV-F1"),
    ]
    m_renkler_h3 = ["#1565C0","#C62828","#2E7D32","#F57F17"]

    for ax, banka in zip(axes, BANKA_SIRASI):
        bl    = BANKA_TR[banka]
        brenk = BANKA_RENK[banka]
        ax.set_facecolor(brenk + "08")

        y_pos = np.arange(len(MODEL_LISTESI))
        bar_h = 0.18

        for mi, (metrik, m_etiket) in enumerate(METRIKLER_H3):
            degerler = []
            for model_adi in MODEL_LISTESI:
                alt = sonuc_df[(sonuc_df["Banka"]==banka) &
                               (sonuc_df["Model"]==model_adi)]
                val = float(alt[metrik].values[0]) \
                    if len(alt)>0 and pd.notna(alt[metrik].values[0]) else 0
                degerler.append(val)

            y_off = y_pos + bar_h * (mi - 1.5)
            ax.barh(y_off, degerler, bar_h * 0.85,
                    color=m_renkler_h3[mi], alpha=0.75,
                    label=m_etiket, edgecolor="white", lw=0.5)

        ax.axvline(H3_ESIK, color="#B71C1C", ls="--", lw=2.5,
                   zorder=5, alpha=0.9)

        acc_vals = []
        for model_adi in MODEL_LISTESI:
            alt = sonuc_df[(sonuc_df["Banka"]==banka) &
                           (sonuc_df["Model"]==model_adi)]
            acc_vals.append(float(alt["Accuracy"].values[0])
                            if len(alt)>0 else 0)

        for yi, (model_adi, acc) in enumerate(zip(MODEL_LISTESI, acc_vals)):
            durum = "✓" if acc >= H3_ESIK else "✗"
            renk_d = "#43A047" if acc >= H3_ESIK else "#E53935"
            ax.text(max(acc_vals)+0.04, yi,
                    f"Acc={acc:.3f}  {durum}",
                    va="center", fontsize=11, fontweight="bold",
                    color=renk_d)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_ISIM[m] for m in MODEL_LISTESI],
                           fontsize=11, fontweight="bold")
        ax.set_xlim([0.30, 1.12])
        ax.set_xlabel("Skor Değeri", fontsize=10)
        ax.set_title(bl, fontsize=13, fontweight="bold", color=brenk)
        ax.legend(loc="lower right", fontsize=9,
                  framealpha=0.9, title="Metrik", title_fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25, ls="--")
        ax.text(H3_ESIK+0.005, -0.6, "H3\n%70",
                fontsize=9, color="#B71C1C", fontweight="bold")

    n_sag = int(sonuc_df["H3_Sagland"].sum())
    n_top = len(sonuc_df)
    fig.text(0.5, -0.04,
             f"Genel Sonuç: {n_sag}/{n_top} model H3 hipotezini sağladı "
             f"({n_sag/n_top*100:.0f}%)",
             ha="center", fontsize=13, fontweight="bold",
             color="#43A047" if n_sag/n_top>=0.7 else "#C62828",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#E8F5E9" if n_sag/n_top>=0.7 else "#FFEBEE",
                       alpha=0.9,
                       edgecolor="#43A047" if n_sag/n_top>=0.7 else "#C62828"))

    plt.tight_layout(w_pad=3)
    plt.savefig("results/figures/27_h3_hipotez_panel.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 27_h3_hipotez_panel.png")

    # ─── FİGÜR 28: CV Boxplot — 3×3 Grid ─────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "Şekil 28 — 5-Kat Çapraz Doğrulama F1 Skoru: Banka × Algoritma\n"
        "Her nokta bir katın F1 değeri  ·  Kutu: çeyrekler arası  ·  "
        "Çizgi: medyan  ·  Sağ kutu: özet istatistikler",
        fontsize=13, fontweight="bold", y=1.01,
    )
    for ci, model_adi in enumerate(MODEL_LISTESI):
        fig.text(0.19 + ci*0.27, 1.005,
                 MODEL_ISIM[model_adi], ha="center",
                 fontsize=14, fontweight="bold",
                 color=MODEL_RENK[model_adi],
                 transform=fig.transFigure)

    for ri, banka in enumerate(BANKA_SIRASI):
        bl    = BANKA_TR[banka]
        brenk = BANKA_RENK[banka]

        for ci, model_adi in enumerate(MODEL_LISTESI):
            ax     = axes[ri, ci]
            ax.set_facecolor("white")
            m_renk = MODEL_RENK[model_adi]

            cv_arr = cv_verileri.get(banka, {}).get(model_adi)
            if cv_arr is None:
                ax.axis("off"); continue
            cv_arr = np.array(cv_arr)

            bp = ax.boxplot([cv_arr], patch_artist=True,
                            medianprops=dict(color="white", lw=3),
                            flierprops=dict(marker="D", ms=5,
                                            markerfacecolor=m_renk,
                                            alpha=0.6),
                            widths=0.45, positions=[0])
            bp["boxes"][0].set_facecolor(m_renk)
            bp["boxes"][0].set_alpha(0.75)
            bp["boxes"][0].set_linewidth(2)
            for w in bp["whiskers"]:
                w.set_color(m_renk); w.set_lw(1.5)
            for c in bp["caps"]:
                c.set_color(m_renk); c.set_lw(2)

            rng = np.random.RandomState(42)
            jit = rng.uniform(-0.12, 0.12, len(cv_arr))
            ax.scatter(jit, cv_arr, s=80, color=m_renk,
                       zorder=5, edgecolors="white", lw=1.5, alpha=0.85)

            mean_v = cv_arr.mean()
            ax.axhline(mean_v, color=m_renk, ls="-.", lw=1.8, alpha=0.6)
            ax.axhline(H3_ESIK, color="#B71C1C", ls="--", lw=1.8, alpha=0.8)

            for j_val, cv_val in zip(jit, cv_arr):
                ax.text(j_val+0.02, cv_val+0.003, f"{cv_val:.3f}",
                        fontsize=7.5, ha="left", va="bottom",
                        color=m_renk, alpha=0.8)

            ax.set_xticks([0])
            ax.set_xticklabels(["5 Kat CV"], fontsize=10)
            ax.set_ylim([0.40, 1.05])
            ax.set_ylabel("F1 Skoru (Ağırlıklı)" if ci==0 else "",
                          fontsize=9)
            ax.set_title(f"{bl}\n{MODEL_ISIM[model_adi]}",
                         fontsize=10, fontweight="bold", color=m_renk)
            ax.text(0.97, 0.97,
                    f"Ort: {cv_arr.mean():.3f}\n"
                    f"Std: {cv_arr.std():.3f}\n"
                    f"Min: {cv_arr.min():.3f}\n"
                    f"Max: {cv_arr.max():.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", alpha=0.92,
                              edgecolor=m_renk, lw=1.5))
            ax.text(0.52, H3_ESIK+0.01, "H3 Eşiği",
                    fontsize=8, color="#B71C1C", fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.25, ls="--")

    plt.tight_layout(h_pad=4, w_pad=2.5)
    plt.savefig("results/figures/28_cv_performans.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 28_cv_performans.png")

    # ─── FİGÜR 29: Radar Grafiği — Koyu Tema ─────────────────
    RADAR_METRIKLER = ["Accuracy","F1_Agirlik","ROC_AUC",
                       "Precision","Recall","CV_F1_Ort"]
    RADAR_ETIKET    = ["Accuracy","F1\nAğırlık","ROC\nAUC",
                       "Precision","Recall","CV\nF1"]
    N_R = len(RADAR_METRIKLER)
    angiller = [n/float(N_R)*2*np.pi for n in range(N_R)]
    angiller += angiller[:1]

    fig, axes_radar = plt.subplots(
        1, 3, figsize=(21, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        "Şekil 29 — Algoritma Çok Boyutlu Performans Karşılaştırması (Radar Grafiği)\n"
        "Her eksen bir metriği temsil eder  ·  Polygon alanı büyüdükçe model daha başarılı  ·  "
        "Sarı kesik çizgi: H3 %70 eşiği",
        fontsize=13, fontweight="bold", color="white", y=1.02,
    )

    for ax_r, banka in zip(axes_radar, BANKA_SIRASI):
        bl    = BANKA_TR[banka]
        brenk = BANKA_RENK[banka]
        ax_r.set_facecolor("#1A1A2E")

        for model_adi in MODEL_LISTESI:
            alt = sonuc_df[(sonuc_df["Banka"]==banka) &
                           (sonuc_df["Model"]==model_adi)]
            if len(alt) == 0:
                continue
            degerler = []
            for metrik in RADAR_METRIKLER:
                val = float(alt[metrik].values[0]) \
                    if pd.notna(alt[metrik].values[0]) else 0
                degerler.append(val)
            degerler += degerler[:1]

            ax_r.plot(angiller, degerler, lw=2.5,
                      color=MODEL_RENK[model_adi],
                      marker=MODEL_MARKER[model_adi],
                      ms=8, markeredgecolor="white",
                      markeredgewidth=1.5,
                      label=MODEL_ISIM[model_adi], zorder=5)
            ax_r.fill(angiller, degerler,
                      color=MODEL_RENK[model_adi], alpha=0.12)

        ax_r.set_xticks(angiller[:-1])
        ax_r.set_xticklabels(RADAR_ETIKET, fontsize=10,
                             color="white", fontweight="bold")
        ax_r.set_ylim([0,1])
        ax_r.set_yticks([0.4, 0.6, 0.7, 0.8, 1.0])
        ax_r.set_yticklabels(["0.4","0.6","0.7","0.8","1.0"],
                             fontsize=7.5, color="#AAA")
        ax_r.plot(angiller, [H3_ESIK]*(N_R+1),
                  ls="--", color="#F57F17", lw=1.5, alpha=0.7, zorder=3)
        ax_r.text(angiller[0], H3_ESIK+0.04,
                  "H3 %70", fontsize=8, color="#F57F17", ha="center")
        ax_r.grid(color="#FFFFFF", alpha=0.15, ls="--")
        ax_r.spines["polar"].set_visible(False)
        ax_r.set_title(bl, fontsize=13, fontweight="bold",
                       color=brenk, pad=20)
        ax_r.legend(loc="lower right",
                    bbox_to_anchor=(1.35, -0.12),
                    fontsize=9, framealpha=0.8,
                    facecolor="#1A1A2E", labelcolor="white",
                    edgecolor="#555")

    plt.tight_layout(pad=2)
    plt.savefig("results/figures/29_radar_algoritma.png",
                dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print("  ✓ 29_radar_algoritma.png")
    """
    # ============================================================
    # 5. TERMİNAL ÖZET
    # ============================================================
    print("\n" + "=" * 65)
    print("SONUÇ TABLOSU")
    print("=" * 65)
    goster = sonuc_df[[
        "Banka_TR","Model","Accuracy","F1_Agirlik",
        "ROC_AUC","CV_F1_Ort","CV_F1_Std","H3_Sagland"
    ]].copy()
    goster.columns = ["Banka","Model","Accuracy","F1_Ağırlık",
                      "ROC-AUC","CV-F1 Ort","CV-F1 Std","H3"]
    print(goster.to_string(index=False))

    print("\n" + "=" * 65)
    print(f"H3 HİPOTEZİ: ≥%{H3_ESIK*100:.0f} Accuracy")
    print("=" * 65)
    for _, row in sonuc_df.iterrows():
        durum = "✓ SAĞLANDI" if row["H3_Sagland"] else "✗ SAĞLANAMADI"
        print(f"  {row['Banka_TR']:12s} | {row['Model']:25s} | "
              f"Acc={row['Accuracy']:.4f}  {durum}")

    n_sag = int(sonuc_df["H3_Sagland"].sum())
    print(f"\n  Toplam: {n_sag}/{len(sonuc_df)} model H3'ü sağladı")

    print(f"""
Üretilen dosyalar:
  results/reports/model_sonuclari.csv
  results/reports/ozellik_onem.csv
  results/figures/23_roc_egrisi.png
  results/figures/24_karisiklik_matrisi.png
  results/figures/25_model_karsilastirma.png
  results/figures/26_ozellik_onem.png
  results/figures/27_h3_hipotez_panel.png
  results/figures/28_cv_performans.png
  results/figures/29_radar_algoritma.png

Sıradaki adım: python adim6_karsilastirma.py
""")