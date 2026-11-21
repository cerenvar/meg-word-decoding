# TEK SEFERLİK · Drive'daki İP-1 dosyalarını İngilizce adlara taşır. Geçici, boş bir not defterinde çalıştırın.
import shutil
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
BASE = Path("/content/drive/MyDrive/MEG_2209A")
OUT = BASE / "ip1"

# 1) Notebook'un kaldığı yerden devam etmek için okuduğu dosyalar: yeni adla taşınır
RESUME = {
    OUT / "kayit_basliklari.csv": OUT / "record_headers.csv",
    OUT / "pencere_indeksi.parquet": OUT / "word_events.parquet",
    OUT / "kalite_raporu.csv": OUT / "quality_report.csv",
    OUT / "normalizasyon_subj0_train.npz": OUT / "normalization_sums_subj0_train.npz",
    BASE / "arsiv_500ms_pT": BASE / "windows_500ms_pT",
}
for old, new in RESUME.items():
    if new.exists() and new.is_dir() and not any(new.iterdir()):
        new.rmdir()
    if old.exists() and not new.exists():
        old.rename(new)
        print("taşındı:", old.name, "→", new.name)
    elif new.exists():
        print("zaten yeni adında:", new.name)
    else:
        print("BULUNAMADI:", old)

# 2) Her çalıştırmada yeniden üretilen eski çıktılar: silinmez, ayrı klasöre kaldırılır
REGENERATED = ["kelime_dizileri.json", "veri_erisim_tablosu.csv", "olay_butunlugu.csv", "zaman_cakismalari.csv",
               "olay_zaman_kaymasi.png", "sinif_frekanslari.csv", "katilimci_rolleri.json",
               "uyarlama_alt_kumeleri.csv", "rol_haritasi.png", "sizinti_testleri.csv", "aykiri_kanallar.csv",
               "pencere_manifesti.parquet", "normalizasyon_subj0_train_parametreleri.npz", "on_olcum.json",
               "sapma_kaydi.csv"]
old_dir = OUT / "_eski_adlar"
old_dir.mkdir(exist_ok=True)
for name in REGENERATED:
    if (OUT / name).exists():
        shutil.move(str(OUT / name), str(old_dir / name))
print("Kenara kaldırılan eski çıktı:", len(list(old_dir.iterdir())))

# 3) Kontrol
required = [OUT / "record_headers.csv", OUT / "word_events.parquet", OUT / "quality_report.csv",
            OUT / "normalization_sums_subj0_train.npz", BASE / "windows_500ms_pT"]
missing = [str(p) for p in required if not p.exists()]
n_archive = sum(1 for _ in (BASE / "windows_500ms_pT").glob("*.npy")) if not missing else 0
print("Eksik:", missing or "yok", "· arşivdeki kayıt dosyası:", n_archive, "/ 226")
assert not missing and n_archive == 226, "Taşıma tamamlanmadı; İP-1 notebook'unu çalıştırmayın"
print("Hazır: İngilizce adlı İP-1 notebook'unu açıp Tümünü çalıştır diyebilirsiniz.")
