# 🏠 House Price Estimation Using CNNs

This project implements a convolutional neural network (CNN) using Keras to predict **house prices** based on image data. Unlike typical CNNs used for classification, this model performs **regression** by outputting a single continuous value.

---

## 📊 Key Characteristics

- Uses a CNN model without a softmax layer
- Replaces softmax with a fully-connected layer + linear activation for regression
- Trains using MAPE (Mean Absolute Percentage Error) loss
- Data preprocessing includes combining four house images into one tiled composite
- House attributes (bedrooms, bathrooms, area, etc.) are used in combination with image input

---

## 📦 Dataset

- 📌 **Source**: 2016 Paper: _"House price estimation from visual and textual features"_
- 📚 GitHub: [emanhamed/Houses-dataset](https://github.com/emanhamed/Houses-dataset)
- 📄 Paper: [arXiv:1609.08399](https://arxiv.org/pdf/1609.08399.pdf)

---

## 🛠️ Model Summary

- Input: 64x64x3 RGB tiled image of four house images
- Conv → ReLU → BatchNorm → MaxPool (×3)
- Flatten → Dense(16) → ReLU → Dropout
- Dense(4) → Dense(1, activation="linear")

---

## 🧪 Training Details

- Loss function: Mean Absolute Percentage Error (MAPE)
- Optimizer: Adam with decay
- 75/25 train/test split
- Normalized prices based on training max
- Achieves reasonable price estimation with less than 60% average MAPE

---

## 🚀 How to Run

1. Download the [Houses-dataset](https://github.com/emanhamed/Houses-dataset)
2. Update `inputPath` and `datasetPath` accordingly
3. Run the notebook or `.py` script using TensorFlow and Keras installed

---

## 📎 Source Tutorial

Based on the official tutorial by PyImageSearch:

🔗 https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/

---

# 🇮🇷 پیش‌بینی قیمت خانه با شبکه‌های عصبی کانولوشنی (CNN)

در این پروژه با استفاده از تصاویر خانه و شبکه CNN، مدلی برای تخمین قیمت خانه‌ها ایجاد شده است. برخلاف مدل‌های معمول که برای دسته‌بندی استفاده می‌شوند، این مدل برای **رگرسیون** طراحی شده است.

---

## ✨ ویژگی‌ها

- حذف لایه softmax و جایگزینی آن با لایه fully-connected + linear activation
- استفاده از خطای میانگین درصد مطلق (MAPE)
- ترکیب ۴ تصویر خانه به یک تصویر تایل شده برای ورودی مدل
- استفاده ترکیبی از داده‌های تصویری و ویژگی‌های عددی مانند متراژ، اتاق و ...

---

## 📚 دیتاست

- مقاله منتشرشده: *House price estimation from visual and textual features* (2016)
- گیت‌هاب: [emanhamed/Houses-dataset](https://github.com/emanhamed/Houses-dataset)
- متن کامل مقاله: [arXiv:1609.08399](https://arxiv.org/pdf/1609.08399.pdf)

---

## 🧠 معماری مدل

- ورودی: تصویر RGB تایل شده ۶۴×۶۴
- ۳ لایه: کانولوشن → ReLU → BatchNorm → Pooling
- سپس Flatten و چند لایه Fully Connected با خروجی یک مقدار قیمت

---

## 🔧 نحوه اجرا

1. دیتاست را از گیت‌هاب دریافت کنید
2. مسیر فایل‌ها را به درستی در کد وارد کنید (`inputPath`, `datasetPath`)
3. اجرا در محیط Keras + TensorFlow

---

## 📎 منبع آموزشی

این پروژه با الهام از آموزش رسمی PyImageSearch ساخته شده است:

🔗 https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/

---

## 🔐 License

Apache License 2.0 — Free for personal and commercial use.
