# Veri açıklaması

Bu belge, ekibin ortak Google Drive klasöründe (`MyDrive/MEG_2209A`) tutulan büyük veri dosyalarını ve İP-1'in çıktılarını açıklar. Bu dosyalar boyutları ve veri setinin lisans koşulları nedeniyle repoda bulunmaz. İP-1'in küçük sonuç tabloları ayrıca repodaki `results/ip1/` klasöründedir.

**İP-1'in çıktıları salt okunurdur.** Sonraki iş paketleri bu dosyaları okur, üzerine yazmaz. İP-1'de değişiklik gerekirse yeni bir sürüm klasörü (`ip1_v2`) üretilir.

## Klasör yapısı

```
MEG_2209A/
├── windows_500ms_pT/               # pencere arşivi (226 dosya, ~28 GB)
└── ip1/                            # İP-1 çıktıları ve ara dosyaları
```

## İP-2'nin kullanacağı dosyalar

Bir modeli eğitmek için gereken her şey şu dört dosyadadır:

| Dosya | Ne işe yarar |
|---|---|
| `ip1/window_manifest.parquet` | Kullanılacak her pencerenin listesi: hangi kayıttan geldiği, etiketi, rolü ve arşivdeki yeri |
| `windows_500ms_pT/*.npy` | Pencerelerin kendisi (MEG sinyali) |
| `ip1/normalization_subj0_train.npz` | Sinyali ölçeklemek için kanal ortalaması ve standart sapması |
| `ip1/participant_roles.json` | Katılımcı grupları ve dışlama gerekçeleri |

### Bir pencereyi yükleme örneği

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/content/drive/MyDrive/MEG_2209A")
manifest = pd.read_parquet(BASE / "ip1" / "window_manifest.parquet")
norm = np.load(BASE / "ip1" / "normalization_subj0_train.npz")

row = manifest.iloc[0]
x = np.load(BASE / "windows_500ms_pT" / row.archive_file, mmap_mode="r")[row.archive_row]   # (306, 125), float16
x = (x.astype(np.float32) - norm["mean"][:, None]) / norm["std"][:, None]               # ölçeklenmiş sinyal
y = int(row.label)                                                                       # 0–49
```

Eğitimde dosyaları Drive'dan doğrudan okumak yavaştır. Arşivi önce Colab diskine kopyalayın; İP-1'in ön ölçümüne göre rastgele sırayla okuma hesaplamadan 50–80 kat yavaş. Hızlı veri yükleyici İP-2'nin ilk işidir.

## Dikkat: iki ayrı kelime/pencere listesi

| Dosya | İçerik | Ne zaman kullanılır |
|---|---|---|
| `ip1/word_events.parquet` | Olay dosyalarındaki **tüm kelimeler**: sözlük dışındakiler, kayda sığmayanlar ve zamanı güvenilmez olanlar dahil | Neredeyse hiç. İP-1'in ara dosyası. |
| `ip1/window_manifest.parquet` | Yalnızca **arşive yazılan** pencereler: 50 kelimelik sözlükten, kayda sığan ve zamanı güvenilir olanlar | **Model eğitimi ve değerlendirmesi için her zaman bu.** |

## `window_manifest.parquet` sütunları

| Sütun | Açıklama |
|---|---|
| `record` | Kayıt kimliği, ör. `sub-1_ses-11_task-Sherlock1_run-1` |
| `subject`, `session`, `task` | Kaydın katılımcısı, oturumu ve görevi (kitap veya korpus) |
| `corpus` | Korpus: `sherlock`, `timit`, `mocha`, `podcasts` |
| `partition` | Veri setinin standart bölümü: `train`, `validation`, `test` |
| `release` | Dosya sürümü: `standart` (tam), `firsthalf` (ilk yarı), `firstquarter` (ilk çeyrek) |
| `participant_role` | Katılımcının projedeki rolü (aşağıdaki tablo) |
| `split_role` | Kaydın projedeki rolü (aşağıdaki tablo) |
| `word` | Kelime (küçük harf; kıvrık kesme işareti düz kesmeye çevrilmiş, ör. `it's`) |
| `label` | 0–49 arası sınıf etiketi; sıra, PNPL'nin resmî sözlüğündeki (`load_vocabulary("primary")`) sıradır |
| `onset` | Kelimenin başlangıç zamanı, saniye (olay dosyasındaki `timemeg` sütunu) |
| `duration` | Kelimenin süresi, saniye |
| `gap_prev`, `gap_next` | Önceki ve sonraki kelimenin başlangıcına uzaklık, saniye |
| `start`, `stop` | Pencerenin kayıttaki başlangıç ve bitiş örneği (250 Hz; `stop − start = 125`) |
| `event_row` | Kelimenin olay dosyasındaki satır numarası (uyaran kimliği) |
| `in_A1`, `in_A2`, `in_A3` | Pencere ilgili uyarlama alt kümesinde mi (yalnız hedef katılımcıların 11. oturumunda) |
| `in_A_all` | Pencere, katılımcının 11. oturumunda erişilebilen uyarlama verisinin tamamında mı. A1'e ulaşamayan subj24'ün betimsel koşulu bu sütunla seçilir; gerçek süresi `adaptation_subsets.csv` dosyasındaki `all_min` sütunundadır. |
| `archive_file`, `archive_row` | Pencerenin arşivdeki dosyası ve satırı |

`duration`, `gap_prev` ve `gap_next`, İP-2'deki zamanlama kontrolleri (yalnız zamanlama referansı, süre grupları) için hazırlanmıştır.

### Roller

**`participant_role`** (katılımcı düzeyi):

| Değer | Katılımcılar | Anlamı |
|---|---|---|
| `source` | subj0 | Kaynak eğitim katılımcısı |
| `evaluation` | subj1–12 | Hipotez testleri yalnızca bu kişilerde yapılır |
| `development` | subj13, 14, 15, 16, 17, 19, 20, 22 | Uyarlama ayarları yalnızca bu kişilerde seçilir |
| `descriptive_a2` | subj18, 21 | Betimsel; A1–A2 düzeyinde raporlanır |
| `descriptive_a1` | subj23–28, 30–32 | Betimsel; yalnız A1 düzeyinde raporlanır |
| `excluded` | subj29 | Hiçbir analizde kullanılmaz; manifestte yer almaz |

**`split_role`** (kayıt düzeyi):

| Değer | Kayıtlar | Kullanım |
|---|---|---|
| `src_train` | subj0'ın 156 eğitim kaydı | Kaynak model eğitimi, normalizasyon, süre grupları |
| `src_validation` | subj0: Sherlock1 ses-11, MOCHA-TIMIT ses-2, TheMoth ses-29 | Kaynak modelde ayar seçimi ve erken durdurma |
| `src_test` | subj0: Sherlock1 ses-12, MOCHA-TIMIT ses-3, TheMoth ses-30 | Kaynak modelin betimsel testi |
| `eval_adapt` | Değerlendirme katılımcılarının 11. oturumu | Uyarlama (A1 ⊂ A2 ⊂ A3) |
| `eval_test` | Değerlendirme katılımcılarının 12. oturumu | **Yalnızca nihai test.** Hiçbir seçimde kullanılmaz. |
| `dev_adapt` | Geliştirme katılımcılarının 11. oturumu | Ayar seçimi için uyarlama (A1 ⊂ A2) |
| `dev_select` | Geliştirme katılımcılarının 12. oturumu | Uyarlama ayarlarının seçimi |
| `desc_adapt` | Betimsel katılımcıların 11. oturumu | Betimsel uyarlama |
| `desc_test` | Betimsel katılımcıların 12. oturumu | Betimsel test; hipotez testine girmez |

## Pencere arşivi (`windows_500ms_pT/`)

* Kayıt başına bir dosya (`<record>.npy`), toplam 226 dosya.
* Her dosyanın boyutu `(pencere sayısı, 306, 125)`: 306 kanal × 125 örnek (0,5 saniye, 250 Hz). Pencere, kelimenin başlangıcından itibaren 0,5 saniyedir.
* Veri tipi `float16`. Birimler: manyetometrelerde **pT**, gradiyometrelerde **pT/cm**. SI biriminden dönüşüm sabit katsayılarla (1e12 ve 1e10) yapılmıştır; veriden hesaplanmadığı için sızıntı yaratmaz.
* Kanal sırası tüm kayıtlarda aynıdır ve HDF5 kayıtlarındaki sıradır (102 manyetometre, 204 gradiyometre; ör. MEG0111 mag, MEG0112 grad, MEG0113 grad, …).
* subj29'un kayıtları boş dizidir (`(0, 306, 125)`).
* Veri ekibinin uyguladığı ön işleme (bozuk kanal işleme, kafa konumu düzeltmesi, SSS, çentik ve bant geçiren filtre, 250 Hz'e alt örnekleme) yeniden uygulanmamıştır.

## Normalizasyon parametreleri

`ip1/normalization_subj0_train.npz`:

| Alan | Açıklama |
|---|---|
| `mean`, `std` | 306 kanalın ortalaması ve standart sapması (pT, pT/cm) |
| `fitted_on` | Hesaplamada kullanılan kayıtlar: subj0'ın 156 eğitim kaydı |
| `unit` | `pT` |

Bu değerler yalnızca kaynak eğitim kayıtlarından hesaplanmıştır.

* **Varsayılan:** Aynı değerler, hedef katılımcılar dahil bütün katılımcılara ve bütün koşullara (A0 dahil) sabit olarak uygulanır. İP-2'nin veri yükleyicisi bu dosyayı doğrudan kullanır.
* **İP-4'te açık soru:** Hedef katılımcılarda, yalnızca uyarlama verisinden hesaplanan kişiye özgü normalizasyonun bu varsayılandan daha iyi olup olmadığı, diğer uyarlama ayarları gibi yalnızca geliştirme katılımcılarında sınanabilir. Varsayılan değiştirilirse sapma kaydına yazılır.

## `ip1/` klasöründeki diğer dosyalar

| Dosya | İçerik |
|---|---|
| `run_info.json` | İP-1'in çalıştırma bilgileri: tarih, kütüphane sürümleri, veri sürümü (Hugging Face commit), tohum |
| `data_access.csv` | Her kayıt için süre, kelime sayısı, pencere sayısı, görülen sınıf sayısı |
| `class_frequencies.csv` | 50 kelimenin bölümlere göre sayısı |
| `adaptation_subsets.csv` | Hedef katılımcılarda A1–A3 düzeylerinin gerçek süre, pencere ve sınıf sayıları |
| `event_integrity.csv` | Her kayıtta kelime zamanlarının denetimi (geriye giden zaman sayısı vb.) |
| `timing_conflicts.csv` | Zamanın geriye gittiği aralıklar; bu aralıklardaki kelimeler kullanılmamıştır |
| `quality_report.csv` | Kayıt düzeyinde sinyal kalitesi (NaN, Inf, düz ve aykırı kanallar, en büyük genlik) |
| `outlier_channels.csv` | Aykırı işaretlenen kanalların gürültü oranları |
| `leakage_tests.csv` | Sekiz sızıntı testinin sonuçları |
| `deviations.csv` | Öneriden sapmalar ve gerekçeleri |
| `benchmark.json` | Koşu süresi ve bellek ön ölçümü |
| `role_map.png`, `event_timing_drift.png` | Şekiller |
| `record_headers.csv`, `word_events.parquet`, `word_sequences.json`, `normalization_sums_subj0_train.npz` | İP-1'in kaldığı yerden devam etmek için tuttuğu ara dosyalar; sonraki iş paketlerinde kullanılmaz |

## Veri sürümü

| Hugging Face deposu | Commit |
|---|---|
| `pnpl/LibriBrain` | `5a7c332b34fc7be329c4df3e527a41f67bfd878f` |
| `pnpl/LibriBrain2` | `78443d024480c66bf340073c69bef50af8d1d335` |

Kütüphane: `pnpl==0.2.0`.
