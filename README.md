# MEG Sinyallerinden Dinlenen Kelimenin Çözümlenmesi

**Decoding Perceived Words from MEG Signals: A Subject-Adaptive and Data-Efficient Deep Learning Framework**

Bu repo, "MEG Sinyallerinden Dinlenen Kelimenin Çözümlenmesi: Kişiye Uyarlanabilir ve Veri Verimliliği Odaklı Bir Derin Öğrenme Yaklaşımı" adlı araştırma projesinin kodlarını ve sonuç tablolarını içerir.

Projede, katılımcıların sesli kitap dinlerken kaydedilen MEG (manyetoensefalografi) sinyallerinden, 50 kelimelik bir sözlük içinde hangi kelimeyi duyduklarını tahmin eden derin öğrenme modelleri geliştirilmektedir. Modeller derin kayıtlı tek bir katılımcıda (subj0) eğitilip yeni katılımcılara birkaç dakikalık veriyle uyarlanmakta; başarının beyin sinyalinden mi, kelime zamanlamasından mı geldiği kontrollü deneylerle sınanmaktadır.

## Proje durumu

| İş paketi | Konu | Durum |
|---|---|---|
| İP-1 | Veri erişimi, veri hattı ve sızıntı denetimi | ✅ Tamamlandı |
| İP-2 | Referans modeller (M0, M1) ve zamanlama kontrolleri | ⏳ Sırada |
| İP-3 | CNN–Transformer hibrit modeli (M2, E5) | — |
| İP-4 | Kişiye uyarlanabilir model ve ayar seçimi (M3) | — |
| İP-5 | Az veriyle uyarlama ve mimari karşılaştırma deneyleri | — |
| İP-6 | Ablasyon, istatistik ve maliyet analizi | — |

## Repo yapısı

```
meg-word-decoding/
├── README.md
├── requirements.txt      # İP-1'in çalıştırıldığı kütüphane sürümleri
├── .gitignore            # büyük veri dosyalarının repoya eklenmesini engeller
├── notebooks/
│   └── ip1_veri_hatti.ipynb
├── results/
│   └── ip1/              # İP-1'in küçük sonuç tabloları
├── configs/              # İP-2'den itibaren deney yapılandırmaları
└── src/                  # İP-2'den itibaren ortak kod
```

## Veri

Çalışmada açık erişimli **LibriBrain100** veri seti kullanılmaktadır (Mantegna vd., 2026). Veri, veri setiyle birlikte yayımlanan `pnpl` kütüphanesi aracılığıyla Hugging Face'ten indirilir.

**Ham veri ve büyük ara dosyalar bu repoda bulunmaz.** Veri setinin lisans koşulları gereği ham MEG verisi paylaşılmaz; pencere arşivi gibi büyük dosyalar (~28 GB) da Git için uygun değildir. Bu dosyalar ekibin ortak Google Drive klasöründe (`MyDrive/MEG_2209A`) tutulur. Klasördeki dosyaların açıklaması, klasörün içindeki `README_veri.md` dosyasındadır.

| Nerede | Ne |
|---|---|
| Bu repo | Kod, not defterleri, küçük sonuç tabloları, belgeler |
| Ortak Drive klasörü | Pencere arşivi, pencere listesi (manifest), ara dosyalar |
| Hugging Face (LibriBrain100) | Ham MEG kayıtları ve kelime olay dosyaları |

### Kullanılan veri sürümü

Sonuçların yeniden üretilebilmesi için tüm indirmeler aşağıdaki commit'lere sabitlenmiştir:

| Hugging Face deposu | Commit |
|---|---|
| `pnpl/LibriBrain` | `5a7c332b34fc7be329c4df3e527a41f67bfd878f` |
| `pnpl/LibriBrain2` | `78443d024480c66bf340073c69bef50af8d1d335` |

## Çalışma ortamı

Tüm adımlar Google Colab'da çalıştırılır.

* **Çalışma zamanı:** Tesla T4 GPU
* **Python:** 3.13
* **Temel kütüphane:** `pnpl==0.2.0`; diğer sürümler `requirements.txt` dosyasında
* **Depolama:** Ortak Google Drive klasörü (not defterlerinde `USE_DRIVE = True`)

Hugging Face erişim anahtarı isteğe bağlıdır. Uzun indirmelerde `HTTP 503` hataları görülürse, Colab'ın *Gizli anahtarlar* bölümüne `HF_TOKEN` adıyla bir okuma anahtarı eklenebilir. Anahtarlar ve diğer gizli bilgiler repoya eklenmez.

## İP-1'i çalıştırma

1. `notebooks/ip1_veri_hatti.ipynb` dosyasını Colab'da açın.
2. *Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU* seçin.
3. *Çalışma zamanı → Tümünü çalıştır* deyin. Drive izin ekranında ortak klasörün sahibi olan hesabı seçin.
4. Bölüm 10'daki kontrol listesinde dört maddenin hepsi `True` olmalıdır.

Not defteri kaldığı yerden devam eder: Drive'daki ara dosyalar mevcutsa uzun adımlar (başlık okuma, pencere arşivi) atlanır. Sıfırdan çalıştırmada ~85–90 GB ham veri indirilir ve işlem birkaç saat sürer.

> **Önemli:** Bölüm 8'i (pencere arşivi) aynı anda yalnızca bir kişi çalıştırmalıdır. Paylaşılan Drive klasörüne yazılan dosyalar, yazan kişinin Drive kotasından düşer.

## İP-1 sonuçları

Sonuç tabloları: `results/ip1/`.

* **Veri erişimi:** subj0'ın 162 kaydının tamamına erişildi (156 eğitim, 3 doğrulama, 3 test); eğitimde 50 kelimelik sözlüğe ait 307.329 pencere var. Hedef katılımcıların verisi kademeli yayımlandığı için öneride tanımlanan B planı (R2) uygulandı.
* **Katılımcı rolleri** (tohum 2209):
  * Değerlendirme: subj1–12 (hipotez testleri)
  * Geliştirme: subj13, 14, 15, 16, 17, 19, 20, 22 (uyarlama ayarlarının seçimi)
  * Betimsel: subj18, 21 (A1–A2) ve subj23–28, 30–32 (A1)
  * Dışlanan: subj29 (kelime zamanları sistematik olarak tutarsız)
* **Uyarlama verisi** 11. oturumdan, iç içe üç düzeyde: A1 = 5 dk, A2 = 10 dk, A3 = oturumun tamamı.
* **Sızıntı denetimi:** 8 otomatik testin hepsi geçti.
* **Veri kalitesi:** NaN, Inf veya düz kanal yok. subj0'ın 7 kaydında tek noktalı zaman hataları temizlendi; işaretlenen 60 aykırı kanal incelendi, hiçbiri çıkarılmadı.
* **Ön ölçüm:** Kaynak eğitimde darboğaz veri okuma (15–20 dk/epoch; hesaplama 0,2–0,3 dk/epoch). İP-2'nin ilk işi hızlı bir veri yükleyicidir.

Öneriden sapmalar ve gerekçeleri `results/ip1/deviations.csv` dosyasındadır.

### `results/ip1/` içeriği

| Dosya | İçerik |
|---|---|
| `run_info.json` | Çalıştırma tarihi, kütüphane sürümleri, veri sürümü, tohum |
| `data_access.csv` | Her kayıt için süre, pencere sayısı, görülen sınıf sayısı |
| `class_frequencies.csv` | 50 kelimenin bölümlere göre sayısı |
| `participant_roles.json` | Katılımcı grupları ve dışlama gerekçeleri |
| `adaptation_subsets.csv` | A1–A3 düzeylerinin gerçek süre ve pencere sayıları |
| `event_integrity.csv`, `timing_conflicts.csv` | Kelime zamanlarının denetimi |
| `quality_report.csv`, `outlier_channels.csv` | Sinyal kalitesi ve aykırı kanallar |
| `leakage_tests.csv` | Sekiz sızıntı testinin sonuçları |
| `normalization_subj0_train.npz` | Yalnız kaynak eğitim kayıtlarından hesaplanan kanal ortalaması ve standart sapması (pT, pT/cm) |
| `benchmark.json` | Koşu süresi ve bellek ön ölçümü |
| `deviations.csv` | Öneriden sapmalar ve gerekçeleri |
| `role_map.png`, `event_timing_drift.png` | Şekiller |

## Çalışma kuralları

Bu kurallar sonuçların geçerliliği için zorunludur:

1. **İP-1'in çıktıları salt okunurdur.** Sonraki iş paketleri İP-1'i yeniden çalıştırmaz, çıktılarını okur. İP-1'de bir değişiklik gerekirse mevcut dosyaların üzerine yazılmaz; yeni bir sürüm klasörü (`ip1_v2`) üretilir.
2. **Değerlendirme katılımcılarının 12. oturumu (`eval_test`) yalnızca nihai test için kullanılır.** Hiçbir ayar, erken durdurma veya model seçimi bu veriyle yapılmaz.
3. **Normalizasyon ve süre grupları yalnızca kaynak eğitim verisinden (`src_train`) hesaplanır.**
4. **PNPL'nin otomatik standardizasyonu (`standardize=True`) ve `WordClassification` etiketleri kullanılmaz.** Standardizasyon verilen tüm kayıtlar üzerinden hesaplandığı için sızıntıya yol açar; etiketler 50 kelimelik sözlüğe göre değil, tüm kelimelere göre atanır.
5. **Her deneyin yapılandırması, tohumu, veri sürümü ve sonucu repoda kayıt altına alınır.**

## Atıf

Veri seti:

> Mantegna, F., Jayalath, D., Elvers, G., Kim, T., Ballyk, B., Fung, A., Cho, S., Kwon, T., Kurth, L., Özdogan, M., Landau, G., Somaiya, P., Voets, N., Woolrich, M., & Parker Jones, O. (2026). LibriBrain100: One hundred hours of broad and deep MEG data for neural speech decoding at scale. arXiv preprint.



## Ekip

<!-- Ekip üyelerini ve iş paketi sorumluluklarını buraya ekleyin. -->

## Lisans

<!-- Proje sonunda kodun yayımlanacağı açık kaynak lisansı buraya eklenecektir. Veri setinin kendi lisans koşulları ayrıca geçerlidir. -->
