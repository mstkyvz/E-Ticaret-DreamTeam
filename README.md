# E-Ticaret-DreamTeam

# DreamTeam

# Ekip Üyeleri

 Ömer Yentür   |   https://www.linkedin.com/in/yentur/
 
 Alper Balbay  |   https://www.linkedin.com/in/alperiox/
 
 Mustafa Yavuz |   https://www.linkedin.com/in/mstkyvz/


## Teknik Genel Bakış

- **AI Modeli**: AIDC-AI/Ovis1.6-Gemma2-9B
- **Frontend**: Streamlit
- **Backend API**: FastAPI
- **Görüntü İşleme**: Pillow
- **Dil Algılama**: papluca/xlm-roberta-base-language-detection

## Screenshots

  ![App Screenshot](https://cdn.discordapp.com/attachments/1272855813370806313/1291259017804386344/Ekran_Resmi_2024-10-03_07.42.11.png?ex=66ff722c&is=66fe20ac&hm=367a3a36b6624701a0e3135d82089d7f9847f3619672ffedc75dad12742f6049&)


## Sistem Gereksinimleri

- Python 3.8+
- CUDA uyumlu GPU (en az 12GB VRAM önerilir)
- 16GB+ RAM
- 50GB+ boş disk alanı

## Bağımlılıklar

Ana bağımlılıklar:
- torch==2.0.0
- transformers==4.30.2
- fastapi==0.95.2
- streamlit==1.22.0
- pillow==9.5.0
- optimum==1.16.0
- uvicorn==0.22.0

Tam bağımlılık listesi için `requirements.txt` dosyasına bakın.

## Kurulum

1. Projeyi klonlayın:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Sanal bir ortam oluşturun ve etkinleştirin:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. Gerekli paketleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

4. CUDA ve cuDNN'in doğru sürümlerinin yüklü olduğundan emin olun.

5. `config.json` dosyasını projenizin kök dizinine yerleştirin ve gerekirse yapılandırmayı düzenleyin.

## Konfigürasyon

`config.json` dosyası aşağıdaki ana bölümleri içerir:

- `api`: API sunucu ayarları
- `model`: AI model parametreleri
- `quantization`: Model kuantizasyon ayarları
- `generation`: Metin üretim parametreleri
- `paths`: Dosya yolları
- `supported_languages`: Desteklenen diller
- `language_map`: Dil eşleştirmeleri

Örnek:

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "model": {
    "checkpoint": "AIDC-AI/Ovis1.6-Gemma2-9B",
    "dtype": "bfloat16",
    "max_length": 2048
  },
  "quantization": {
    "method": "qfloat8"
  },
  ...
}
```

## API Endpoints

### POST /process/

Görüntü ve metin işleme için kullanılır.

**Parametreler:**
- `image`: UploadFile - İşlenecek görüntü dosyası
- `text`: str - Görüntüyle ilgili açıklama
- `lang`: str - İstenilen çıktı dili

**Yanıt:**
```json
{
  "output": "Generated text based on the image and input"
}
```

## Model Optimizasyonu

Proje, model performansını artırmak için çeşitli optimizasyon teknikleri kullanır:

1. **Kuantizasyon**: `qfloat8` yöntemi kullanılarak model ağırlıkları kuantize edilir.
2. **CUDA Optimizasyonu**: Model, CUDA-uyumlu GPU'larda çalışacak şekilde optimize edilmiştir.
3. **Bellek Yönetimi**: Gereksiz tensörler temizlenir ve CUDA önbelleği boşaltılır.

## Hata Ayıklama

Yaygın hatalar ve çözümleri:

1. CUDA hatası: CUDA sürümünüzün PyTorch sürümüyle uyumlu olduğundan emin olun.
2. Bellek yetersizliği: Daha büyük VRAM'e sahip bir GPU kullanın veya batch size'ı azaltın.
3. Model yükleme hatası: İnternet bağlantınızı kontrol edin ve model dosyalarının doğru konumda olduğundan emin olun.

## Performans Optimizasyonu

- Büyük dosyalar için önbellek kullanın.
- API isteklerini asenkron olarak işleyin.
- Görüntü ön işleme adımlarını optimize edin.

## Güvenlik Notları

- API'yi public internete açmadan önce uygun kimlik doğrulama ve yetkilendirme mekanizmalarını ekleyin.
- Kullanıcı girdilerini her zaman doğrulayın ve sterilize edin.
- Hassas bilgileri (API anahtarları, veritabanı kimlik bilgileri vb.) çevresel değişkenlerde veya güvenli bir yapılandırma yönetimi sisteminde saklayın.


## Lisans

Bu proje [MIT lisansı](https://opensource.org/licenses/MIT) altında lisanslanmıştır.
