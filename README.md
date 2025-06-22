---
title: Ghibli Image Classifier
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
short_description: A DL model that classifies whether an image is Ghibli or not
---

# Ghibli Image Classifier

A deep learning model that classifies whether an input image is a Ghibli-style image or an original one.  
It uses a fine-tuned ResNet18 model and offers a web interface using Gradio.

Try it here: [Live Demo](https://your-huggingface-space-link)

---

## 🧠 Model

- Pretrained model: `ResNet18` from `torchvision`
- Only the final layer was fine-tuned
- Trained using `CrossEntropyLoss` and `Adam` optimizer

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/Ghibli_Classifier.git
cd Ghibli_Classifier
pip install -r requirements.txt
python app.py
