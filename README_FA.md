# 🧠 آزمایشگاه و مجموعه پژوهشی هوش مصنوعی و یادگیری عمیق
### *بنچ‌مارک‌های جامع، معماری‌های چندوجهی، بینایی ماشین و پردازش تصویر پزشکی*

<div align="center" dir="rtl">

[![GitHub Repo](https://img.shields.io/badge/GitHub-ebi19912%2FAI-181717?style=for-the-badge&logo=github)](https://github.com/ebi19912/AI)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%20%7C%20Vision-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br/>

**[ 🇬🇧 English Documentation (مستندات انگلیسی) ](README.md) | [ 🇮🇷 راهنمای فارسی ](#-درباره-پروژه)**

</div>

---

<div dir="rtl">

## 📌 درباره پروژه

این مخزن یک **مرجع و آزمایشگاه جامع تحقیقاتی در زمینه یادگیری عمیق (Deep Learning) و هوش مصنوعی** است که توسط **[روح‌اله ابراهیمی](https://www.linkedin.com/in/rouhalah-ebrahimi/)** طراحی و توسعه یافته است. این مجموعه شامل بیش از ۵۰ نوت‌بوک تحقیقاتی بر روی مسائلی نظیر بینایی ماشین در پزشکی، طبقه‌بندی چندبرچسبی، سگمنتیشن تصاویر زیست‌پزشکی، شبکه‌های مولد (GANs)، اتوانکودرها، بینایی هوایی و تشخیص اشیاء (YOLOv8)، امنیت بیومتریک و پیش‌بینی سری‌های زمانی مالی است.

### 🔬 محورهای اصلی پژوهش
۱. **پردازش تصویر و تشخیص پزشکی (Medical Diagnostics)**: تشخیص و طبقه‌بندی ۱۴ ناهنجاری ریوی بر روی پایگاه داده استاندارد **NIH ChestX-ray14**، سگمنتیشن ضایعات پوستی با **U-Net** و معماری‌های ترکیبی **Vision Transformer (ViT)**.
۲. **هوش مصنوعی توضیح‌پذیر (Explainable AI - XAI)**: استخراج و نمایش نقشه‌های ویژگی لایه‌ها با قلاب‌های مستقیم در PyTorch و Keras.
۳. **بینایی هوایی و تشخیص اشیاء (Drone Vision & Object Detection)**: بهینه‌سازی مدل‌های پیشرفته **YOLOv8** (از نسخه سبک Nano تا مدل قدرتمند Extra-Large) بر روی دیتاست‌های نظارت پهپادی و **VisDrone2019-DET**.
۴. **مدل‌های مولد و اتوانکودرها (Generative Models & Autoencoders)**: پیاده‌سازی انواع شبکه‌های **DCGAN**، **SGAN** (نیمه‌نظارتی)، **CGAN** (شرطی) و اتوانکودرهای بازسازی و رنگ‌آمیزی تصاویر سیاه‌وسفید.
۵. **امنیت بیومتریک (Biometrics)**: سامانه کامل ثبت، استخراج ویژگی و احراز هویت اثر انگشت بر پایه شبکه‌های کانولوشنی روی پایگاه داده استاندارد **FVC2000**.
۶. **پیش‌بینی سری‌های زمانی مالی (Financial Time-Series)**: پیش‌بینی قیمت ارزهای دیجیتال (BTC/IRT و BNB/IRT) و شاخص بورس داوجونز با شبکه‌های **GRU** و **LSTM** همراه با بهینه‌سازی هایپرپارامترها با **Keras Tuner**.
۷. **امنیت سایبری و یادگیری ماشین (Tabular ML)**: دسته‌بندی استحکام گذرواژه‌ها با مدل‌های یادگیری ماشین و تکنیک‌های ترکیبی Stacking.

---

## 🏗️ ساختار فنی و حوزه‌های تخصصی

### ۱. 🫁 طبقه‌بندی چندبرچسبی تصاویر پزشکی (Chest X-Ray Diagnostics)

#### 🎯 چالش‌های اساسی در داده‌های رادیولوژی قفسه سینه
- **عدم تعادل شدید بین کلاس‌ها**: رخداد برخی بیماری‌ها مانند فتق دیافراگم (Hernia) کمتر از ۰.۵٪ کل داده‌ها است.
- **جلوگیری از نشت داده بیمار (Patient Data Leakage Prevention)**: به علت وجود چندین تصویر رادیولوژی از یک بیمار، داده‌ها به جای تفکیک تصادفی تصویر، بر اساس شناسه بیمار (`Patient_ID`) جداسازی شده‌اند:
  $$	ext{Patients}(\mathcal{D}_{	ext{train}}) \cap 	ext{Patients}(\mathcal{D}_{	ext{val}}) \cap 	ext{Patients}(\mathcal{D}_{	ext{test}}) = \emptyset$$
- **هم‌پوشانی چندبرچسبی (Multi-Label Co-occurrence)**: یک بیمار ممکن است همزمان به چندین عارضه مبتلا باشد.

#### ⚖️ تابع هزینه انتروپی متقاطع دودویی وزن‌دار (Weighted Binary Cross-Entropy)
برای جلوگیری از سوگیری مدل به سمت کلاس‌های منفی با تعداد بالا، تابع هزینه سفارشی زیر پیاده‌سازی شده است:
$$\mathcal{L}_{	ext{WBCE}} = -rac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ w_{p, c} \cdot y_{i, c} \log(\hat{y}_{i, c}) + w_{n, c} \cdot (1 - y_{i, c}) \log(1 - \hat{y}_{i, c}) ight]$$
که در آن وزن‌های مثبت و منفی بر اساس فرکانس وقوع بیماری تعیین می‌شوند:
$$w_{p, c} = rac{N - N_c}{N}, \quad w_{n, c} = rac{N_c}{N}$$

#### 📊 مدل‌های بررسی‌شده در بنچ‌مارک:
- **معماری‌های مدرن ConvNeXt**: شامل تمامی نسخه‌های `Tiny`, `Small`, `Base`, `Large`, `XLarge`
- **شبکه‌های متراکم**: `DenseNet-169`, `DenseNet-201`
- **شبکه‌های باقیمانده (Residual)**: `ResNet-50`, `ResNet-101V2`, `ResNet-152`
- **مدل‌های سبک و بهینه**: `EfficientNetV2-L`, `MobileNetV1`, `MobileNetV2`, `NASNetMobile`
- **مدل‌های پایه کلاسیک**: `VGG-16`, `VGG-19`

#### 🧬 مدل‌های ترکیبی (Hybrid Fusion) و ترنسفورمرهای بینایی
- **`VIT + ConvNeXtBase + VGG16`**: تلفیق ترنسفورمرهای بینایی (Vision Transformer برای استخراج توجه سراسری)، ConvNeXt (کانولوشن‌های عمیق 7x7) و VGG16 (استخراج ویژگی‌های پایه لبه)، آموزش‌دیده با **PyTorch Automatic Mixed Precision (AMP)** و تفکیک کامل شناسه بیماران.
- **`ConvNeXtBase + VGG16`**: شبکه فیوژن ویژگی برای سرطان ریه و بیماری‌های قفسه سینه.
- **`6-Model Ensemble API`**: معماری آنسامبل یکپارچه با ترکیب ۶ مدل مختلف (VGG16, VGG19, ResNet50, InceptionV3, DenseNet201, MobileNet).

---

### ۲. 🔬 سگمنتیشن معنایی ضایعات پوستی (U-Net Skin Lesion Segmentation)

پیاده‌سازی‌شده در فایل [`skin_lesion_segmentation_using_unet.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/skin_lesion_segmentation_using_unet.ipynb):
- **معماری**: شبکه U-Net با مسیر انقباضی (انکودر)، باتل‌نک، مسیر انبساطی (دیکودر) و اتصالات میان‌بر (Skip Connections).
- **تابع هزینه سفارشی ژاکارد (Jaccard Distance Loss)**:
  $$J(y, \hat{y}) = rac{\sum (y \cdot \hat{y}) + \epsilon}{\sum y + \sum \hat{y} - \sum (y \cdot \hat{y}) + \epsilon}, \quad \mathcal{L}_{	ext{Jaccard}} = 1 - J(y, \hat{y})$$
- **معیارهای ارزیابی**: ضریب شباهت دایس (Dice Coefficient)، ضریب اشتراک ژاکارد (IoU)، دقت، صحت، بازخوانی.
- **بهبود نهایی تصویر**: فیلترهای پس‌پردازش برای صاف‌سازی مرزهای ضایعه ملانوما.

---

### ۳. 🔍 هوش مصنوعی توضیح‌پذیر و مصورسازی لایه‌ها (Explainable AI - XAI)

پیاده‌سازی‌شده در فایل [`check_Models_Layers_For_X_ray_Images_.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/check_Models_Layers_For_X_ray_Images_.ipynb):
- تحلیل لایه به لایه شبکه‌ها با قلاب‌های PyTorch و لایه‌های میانی Keras.
- مصورسازی نحوه تبدیل ویژگی‌های سطح پایین (لبه‌ها و گرادیان‌ها) به ساختارهای کالبدی (دنده‌ها و قلب) و الگوهای پاتولوژیک پیچیده.

---

### ۴. 🚁 تشخیص اشیاء هوایی و نظارت پهپادی (Drone Vision)

پیاده‌سازی‌شده در نوت‌بوک‌های [`VisDrone_Detection.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/VisDrone_Detection.ipynb) و [`YOLO_Drone_Detection.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/YOLO_Drone_Detection.ipynb):
- **مدل‌ها**: نسخه‌های سبک **YOLOv8n** و قدرتمند **YOLOv8x**.
- **مجموعه داده‌ها**: دیتاست استاندارد **VisDrone2019-DET** و دیتاست‌های تخصصی رهگیری پرنده‌های هدایت‌پذیر (UAV).
- **آموزش با رزولوشن بالا**: پشتیبانی از ابعاد تصویر تا $1500 	imes 1500$ پیکسل برای شناسایی اهداف بسیار کوچک در فواصل هوایی دور.

---

### ۵. 🎨 شبکه‌های مولد و پردازش تصویر (GANs & Colorization)

پیاده‌سازی‌شده در نوت‌بوک‌های [`All_gan.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/All_gan.ipynb)، [`Colorized.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Colorized.ipynb) و [`Image_Proc_Autoencoder.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Image_Proc_Autoencoder.ipynb):
- **DCGAN**: شبکه مولد رقابتی کانولوشنی با توابع فعال‌سازی LeakyReLU و کانولوشن‌های معکوس.
- **SGAN**: مدل مولد نیمه‌نظارتی برای استفاده همزمان از داده‌های برچسب‌دار و بدون‌برچسب.
- **CGAN**: مدل مولد شرطی برای تولید تصاویر کنترل‌شده با شناسه کلاس.
- **اتوانکودرهای رنگ‌آمیزی**: تبدیل تصاویر سیاه‌وسفید تک‌کاناله به خروجی‌های سه‌کاناله RGB روی دیتاست‌های چهره افراد مشهور (CelebA)، گل‌ها و مناظر طبیعی.

---

### ۶. 🔐 امنیت بیومتریک و شناسایی اثر انگشت

پیاده‌سازی‌شده در نوت‌بوک‌های [`Fingerprint_CNN_English.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_English.ipynb) و [`Fingerprint_CNN_Persian.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_Persian.ipynb):
- مجموعه داده استاندارد بین‌المللی **FVC2000 DB4_B**.
- استخراج بردار ویژگی کانولوشنی از الگوهای اثر انگشت.
- توابع کامل ثبت اثر انگشت (`register_fingerprint`) و احراز هویت بر اساس فاصله شباهت (`authenticate_fingerprint`).

---

### ۷. 📈 پیش‌بینی سری‌های زمانی مالی و ارزهای دیجیتال

پیاده‌سازی‌شده در نوت‌بوک‌های [`BTCIRT_Predict.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/BTCIRT_Predict.ipynb)، [`BNBIRT_Predict.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/BNBIRT_Predict.ipynb) و [`DJPrediction.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/DJPrediction.ipynb):
- پیش‌بینی نرخ جفت ارزهای بیت‌کوین/ریال (BTC/IRT) و بایننس کوین/ریال (BNB/IRT) با شبکه‌های بازگشتی GRU.
- بهینه‌سازی خودکار هایپرپارامترها (تعداد لایه‌ها، دراپ‌اوت و نرخ یادگیری) با **Keras Tuner**.
- پیش‌بینی قیمت شاخص داوجونز بورس آمریکا با شبکه‌های LSTM و GRU.

---

### ۸. 🛡️ امنیت سایبری و یادگیری ماشین جدولی

پیاده‌سازی‌شده در فایل [`iPassword_(fin).ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/iPassword_(fin).ipynb):
- مهندسی ویژگی بر روی طول گذرواژه‌ها، انتروپی کاراکترها و نسبت علائم خاص.
- مقایسه جامع الگوریتم‌های **XGBoost, Random Forest, AdaBoost, Decision Tree, Logistic Regression, KNN و Stacking Classifier**.

</div>

---

## 📂 فهرست کامل فایل‌ها و لینک‌های اجرای مستقیم در Google Colab

| دسته‌بندی | نام فایل نوت‌بوک | معماری و روش اصلی | مجموعه داده | اجرای مستقیم در Colab |
| :--- | :--- | :--- | :--- | :---: |
| **Medical Vision** | `ConvNeXtBase_ChestClassification_wbce.ipynb` | تشخیص چندبرچسبی با ConvNeXt-Base | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtBase_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtTiny_ChestClassification_wbce.ipynb` | تشخیص چندبرچسبی با ConvNeXt-Tiny | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtTiny_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtSmall_ChestClassification_wbce.ipynb` | تشخیص چندبرچسبی با ConvNeXt-Small | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtSmall_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtLarge_ChestClassification_wbce.ipynb` | تشخیص چندبرچسبی با ConvNeXt-Large | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtLarge_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtXLarge_ChestClassification_wbce.ipynb` | تشخیص چندبرچسبی با ConvNeXt-XLarge | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtXLarge_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `DenseNet169_ChestClassification_wbce.ipynb` | تشخیص با شبکه متراکم DenseNet-169 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DenseNet169_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `DenseNet201_ChestClassification_wbce.ipynb` | تشخیص با شبکه متراکم DenseNet-201 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DenseNet201_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `DenseNet201__ChestClassification_wbce.ipynb` | تشخیص با DenseNet-201 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DenseNet201__ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ResNet101V2_ChestClassification_wbce.ipynb` | تشخیص با ResNet-101V2 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ResNet101V2_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ResNet152_ChestClassification_wbce.ipynb` | تشخیص با ResNet-152 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ResNet152_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `EfficientNetV2L_ChestClassification_wbce.ipynb` | تشخیص با EfficientNetV2-L | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/EfficientNetV2L_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `MobileNet_ChestClassification_wbce.ipynb` | تشخیص با MobileNetV1 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNet_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `MobileNetV2_ChestClassification_wbce.ipynb` | تشخیص با MobileNetV2 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNetV2_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `NASNetMobile_ChestClassification_wbce.ipynb` | تشخیص با NASNetMobile | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/NASNetMobile_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `VGG16_ChestClassification_wbce.ipynb` | تشخیص با VGG-16 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VGG16_ChestClassification_wbce.ipynb) |
| **Hybrids & Fusion** | `VIT+ConvNeXtBase+VGG16_Chest_final.ipynb` | مدل فیوژن ترنسفورمر، کانونکست و VGG16 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VIT%2BConvNeXtBase%2BVGG16_Chest_final.ipynb) |
| **Hybrids & Fusion** | `ConvNeXtBase+VGG16_ChestClassification_wbce_Full.ipynb` | مدل ترکیبی ConvNeXt و VGG16 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtBase%2BVGG16_ChestClassification_wbce_Full.ipynb) |
| **Hybrids & Fusion** | `Lung_Cancer_ConvNeXtBase+VGG16_ChestClassification_wbce_Full.ipynb` | مدل ترکیبی سرطان ریه | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Lung_Cancer_ConvNeXtBase%2BVGG16_ChestClassification_wbce_Full.ipynb) |
| **Hybrids & Fusion** | `Copy_of_Lung_Cancer_ConvNeXtBase+VGG16_ChestClassification_wbce_Full.ipynb` | مدل ترکیبی سرطان ریه | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_Lung_Cancer_ConvNeXtBase%2BVGG16_ChestClassification_wbce_Full.ipynb) |
| **Hybrids & Fusion** | `MobileNet+VGG16_ChestClassification_wbce.ipynb` | مدل ترکیبی MobileNet و VGG16 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNet%2BVGG16_ChestClassification_wbce.ipynb) |
| **Hybrids & Fusion** | `API_model_with_six_models_Copy.ipynb` | آنسامبل ۶ مدل یادگیری عمیق | Chest X-Ray | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/API_model_with_six_models_Copy.ipynb) |
| **Segmentation** | `skin_lesion_segmentation_using_unet.ipynb` | سگمنتیشن ضایعات پوستی با U-Net | ISIC Skin Lesion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/skin_lesion_segmentation_using_unet.ipynb) |
| **Explainable AI** | `check_Models_Layers_For_X_ray_Images_.ipynb` | استخراج نقشه‌های ویژگی لایه‌ها با Hook | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/check_Models_Layers_For_X_ray_Images_.ipynb) |
| **Explainable AI** | `Copy_of_check_Models_Layers_For_X_ray_Images_.ipynb` | استخراج نقشه‌های ویژگی لایه‌ها | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_check_Models_Layers_For_X_ray_Images_.ipynb) |
| **Autoencoders** | `AutoEncoder_ConvNeXtBase_ChestClassification_wbce.ipynb` | اتوانکودر بازسازی با ConvNeXt-Base | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/AutoEncoder_ConvNeXtBase_ChestClassification_wbce.ipynb) |
| **Autoencoders** | `AutoEncoder_Vgg16_ChestXray_NIH.ipynb` | اتوانکودر بازسازی با VGG-16 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/AutoEncoder_Vgg16_ChestXray_NIH.ipynb) |
| **Autoencoders** | `AutoEncoder_vgg19_ChestClassification_wbce.ipynb` | اتوانکودر بازسازی با VGG-19 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/AutoEncoder_vgg19_ChestClassification_wbce.ipynb) |
| **Autoencoders** | `autoencoder_vgg19_chestxray_nih.ipynb` | اتوانکودر بازسازی با VGG-19 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/autoencoder_vgg19_chestxray_nih.ipynb) |
| **Autoencoders** | `vgg19_autoencoder_chest_xray_auc.ipynb` | ارزیابی AUC اتوانکودر VGG-19 | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/vgg19_autoencoder_chest_xray_auc.ipynb) |
| **Autoencoders** | `Autoencoder_chest_xray_Full.ipynb` | اتوانکودر کامل تصویر قفسه سینه | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Autoencoder_chest_xray_Full.ipynb) |
| **Autoencoders** | `Copy_of_chest_xray_Autoencoder.ipynb` | اتوانکودر تصویر قفسه سینه | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_chest_xray_Autoencoder.ipynb) |
| **Autoencoders** | `MobileNetVGG16Autoencoder_DffrntLoss_ChestMultiLBL.ipynb` | اتوانکودر دوگانه با چند تابع هزینه | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNetVGG16Autoencoder_DffrntLoss_ChestMultiLBL.ipynb) |
| **Drone Detection** | `VisDrone_Detection.ipynb` | تشخیص اشیاء هوایی با YOLOv8n روی VisDrone | VisDrone2019-DET | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VisDrone_Detection.ipynb) |
| **Drone Detection** | `YOLO_Drone_Detection.ipynb` | شناسایی و رهگیری پهپاد با YOLOv8n | Aerial Drone Dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/YOLO_Drone_Detection.ipynb) |
| **Drone Detection** | `YOLO_Drone_Detection_2.ipynb` | شناسایی دقیق پهپاد با YOLOv8x | Aerial Drone Dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/YOLO_Drone_Detection_2.ipynb) |
| **GANs & Processing** | `All_gan.ipynb` | پیاده‌سازی مدل‌های DCGAN، SGAN و CGAN | MNIST / Synthetic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/All_gan.ipynb) |
| **GANs & Processing** | `Colorized.ipynb` | اتوانکودر رنگ‌آمیزی تصاویر سیاه‌وسفید | CelebA / Landscapes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Colorized.ipynb) |
| **GANs & Processing** | `colorization_autoencoder.ipynb` | اتوانکودر تبدیل خاکستری به RGB | Celebrity Faces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/colorization_autoencoder.ipynb) |
| **GANs & Processing** | `colorized_image.ipynb` | رنگ‌آمیزی تصاویر گل‌ها با شبکه عصبی | Flowers Dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/colorized_image.ipynb) |
| **GANs & Processing** | `Image_Proc_Autoencoder.ipynb` | اتوانکودر بازسازی پلاک خودرو و تصاویر رودخانه | Multimodal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Image_Proc_Autoencoder.ipynb) |
| **GANs & Processing** | `VGG19_Autoencoder.ipynb` | اتوانکودر استخراج فضای نهان با VGG-19 | Custom Image Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VGG19_Autoencoder.ipynb) |
| **Biometrics** | `Fingerprint_CNN_English.ipynb` | سامانه ثبت و احراز هویت اثر انگشت (انگلیسی) | FVC2000 DB4_B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_English.ipynb) |
| **Biometrics** | `Fingerprint_CNN_Persian.ipynb` | سامانه ثبت و احراز هویت اثر انگشت (فارسی) | FVC2000 DB4_B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_Persian.ipynb) |
| **Signal Processing** | `PatternRecognition4th_2009_6_Chapter_Final.ipynb` | تبدیلات کسینوسی، سینوسی و موجک هار | Image Benchmarks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/PatternRecognition4th_2009_6_Chapter_Final.ipynb) |
| **Time-Series** | `BTCIRT_Predict.ipynb` | پیش‌بینی قیمت بیت‌کوین/تومان با GRU | BTC/IRT Historical | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/BTCIRT_Predict.ipynb) |
| **Time-Series** | `BNBIRT_Predict.ipynb` | پیش‌بینی بایننس‌کوین با Keras Tuner | BNB/IRT Historical | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/BNBIRT_Predict.ipynb) |
| **Time-Series** | `DJPrediction.ipynb` | پیش‌بینی شاخص سهام داوجونز با LSTM و GRU | Dow Jones Index | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DJPrediction.ipynb) |
| **Cybersecurity** | `iPassword_(fin).ipynb` | سنجش استحکام رمز عبور با الگوریتم‌های یادگیری ماشین | Password Strength Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/iPassword_(fin).ipynb) |
| **Cybersecurity** | `Copy_of_Password0.ipynb` | سنجش استحکام رمز عبور | Password Strength Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_Password0.ipynb) |
| **GUI & Simulation** | `DiceGameSimulatorGUI.ipynb` | شبیه‌سازی آماری بازی تاس با رابط گرافیکی Tkinter | Synthetic Simulation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DiceGameSimulatorGUI.ipynb) |
| **Comparative Analytics** | `Untitled1.ipynb` | تحلیل و کاوش مدل‌های رادیولوژی | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Untitled1.ipynb) |
| **Comparative Analytics** | `Untitled2.ipynb` | تحلیل مقایسه‌ای مدل‌ها بر روی داده‌های بیماران | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Untitled2.ipynb) |
| **Comparative Analytics** | `Untitled4.ipynb` | ترسیم نمودارهای مقایسه‌ای معیار F1-Score | Model Benchmark Metrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Untitled4.ipynb) |

---

<div dir="rtl">

## 🚀 راهنمای نصب و اجرا

### ۱. اجرای آنلاین بر روی Google Colab (پیشنهادی)
۱. بر روی نشان **Open in Colab** نوت‌بوک مد نظر خود در جدول بالا کلیک کنید.
۲. در محیط کولب از منوی **Runtime > Change runtime type** گزینه **GPU (T4 یا A100)** را انتخاب کنید.
۳. در صورت نیاز به دانلود دیتاست از کگل، فایل توکن `kaggle.json` خود را هنگام درخواست بارگذاری نمایید.

### ۲. اجرای محلی (Local Environment)

```bash
# ۱. کلون کردن مخزن
git clone https://github.com/ebi19912/AI.git
cd AI

# ۲. ساخت محیط مجازی
python -m venv venv

# فعال‌سازی در ویندوز:
.\venv\Scripts\activate
# فعال‌سازی در لینوکس / مک:
source venv/bin/activate

# ۳. نصب کتابخانه‌های مورد نیاز
pip install tensorflow torch torchvision torchaudio ultralytics opencv-python scikit-learn pandas numpy matplotlib seaborn jupyter keras-tuner xgboost
```

---

## 👤 نویسنده و توسعه‌دهنده

**روح‌اله ابراهیمی**
- **گیت‌هاب**: [@ebi19912](https://github.com/ebi19912)
- **لینکدین**: [Rouhalah Ebrahimi](https://www.linkedin.com/in/rouhalah-ebrahimi/)
- **آدرس مخزن**: [https://github.com/ebi19912/AI](https://github.com/ebi19912/AI)

---

## 📜 مجوز (License)
این پروژه تحت مجوز متن‌باز [MIT License](LICENSE) منتشر شده است.

</div>
