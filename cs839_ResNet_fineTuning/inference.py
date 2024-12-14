from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
import numpy as np
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image 
import PIL 
import matplotlib.pyplot as plt

repo_name = "/afs/cs.wisc.edu/u/b/z/bzou/private/cs839/resnet_finetuned/"

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained(repo_name)

#dataset = load_dataset("/afs/cs.wisc.edu/u/b/z/bzou/private/cs839/Medical_MNIST3/test/")
dataset = load_dataset("/afs/cs.wisc.edu/u/b/z/bzou/private/cs839/Brain_Tumor_MRI_Dataset/")
#example = dataset["train"][0]
#print(example)
#plt.imshow(example['image'])
#plt.axis("off")
#plt.show()

labels = dataset["test"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
    



normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

val_transforms  = Compose([
    Resize((image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])),  # Resize images to match ResNet input
    ToTensor(),
    normalize,  # Normalize images
])

def preprocess_val(example_batch):
    #print("preprocess_val")
    #print(example_batch)
    #example["pixel_values"] = transform(example["image"])
    example_batch['pixel_values'] = [
        val_transforms(image.convert("RGB")) for image in example_batch['image']
    ]
    return example_batch
    
#print("dataset structure")
#print(dataset)
test_dataset=dataset["test"]
test_dataset.set_transform(preprocess_val)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
    
a = collate_fn(test_dataset)
pixel_values,labels = a["pixel_values"], a["labels"]
print("pixel_values")
print(pixel_values.shape)
print("labels")
print(labels.shape)

import torch

# forward pass
with torch.no_grad():
    total = 0;
    acc = 0;
    outputs = model(pixel_values)
    print(outputs)
    logits = outputs.logits
    print(logits.shape)
    for logit,label in zip(logits,labels):
        predicted_class_idx = logit.argmax(-1).item()
        #print(label)
        #print(predicted_class_idx)
        if predicted_class_idx==label.item():
            acc+=1;
        total+=1;
print(acc/total)