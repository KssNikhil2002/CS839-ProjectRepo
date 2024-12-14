from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

from PIL import Image 
import PIL 
import matplotlib.pyplot as plt

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
plt.imshow(image)
plt.axis("off")
plt.show()

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(predicted_label)
print(model.config.id2label[predicted_label])
#https://huggingface.co/microsoft/resnet-50