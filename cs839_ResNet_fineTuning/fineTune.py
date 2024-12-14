#reference:https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb
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

from PIL import Image 
import PIL 
import matplotlib.pyplot as plt

#dataset = load_dataset("/afs/cs.wisc.edu/u/b/z/bzou/private/cs839/Medical_MNIST2/")
dataset = load_dataset("/afs/cs.wisc.edu/u/b/z/bzou/private/cs839/Brain_Tumor_MRI_Dataset/")
#example = dataset["train"][0]
#print(example)
#plt.imshow(example['image'])
#plt.axis("off")
#plt.show()

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
    
print(id2label[2])

from transformers import AutoModelForImageClassification

model_name = "microsoft/resnet-50"
image_processor  = AutoImageProcessor.from_pretrained(model_name)
print(image_processor)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms  = Compose([
    Resize((image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])),  # Resize images to match ResNet input
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,  # Normalize images
])

val_transforms  = Compose([
    Resize((image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])),  # Resize images to match ResNet input
    ToTensor(),
    normalize,  # Normalize images
])

# Apply transformations
def preprocess_train(example_batch):
    #print("preprocess_train")
    #print(example_batch)
    #example["pixel_values"] = transform(example["image"])
    example_batch['pixel_values'] = [
        train_transforms(image.convert("RGB")) for image in example_batch['image']
    ]
    return example_batch

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

#train_dataset = dataset["train"]
#eval_dataset=dataset["validation"]
#train_dataset.set_transform(preprocess_train)
#eval_dataset.set_transform(preprocess_val)
#print("after_transform")
#print(train_dataset)
#print(eval_dataset)
#print(train_dataset[0])
#print(dataset)
splits = dataset["train"].train_test_split(test_size=0.1)
train_dataset = splits['train']
eval_dataset = splits['test']
train_dataset.set_transform(preprocess_train)
eval_dataset.set_transform(preprocess_val)

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes = True,
)
from transformers import TrainingArguments, Trainer
import torch

# Define training arguments
training_args = TrainingArguments(
    output_dir="./resnet_finetuned",#
    remove_unused_columns=False,
    evaluation_strategy="epoch",#
    save_strategy="epoch",#
    per_device_train_batch_size=16,#
    per_device_eval_batch_size=16,#
    num_train_epochs=10,#
    learning_rate=5e-5,#
    weight_decay=0.01,
    logging_dir="./logs",#
    logging_steps=10,#
    load_best_model_at_end=True,#
    metric_for_best_model="accuracy",#
)

# Define metric
from datasets import load_metric
metric = load_metric("accuracy")

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = torch.argmax(torch.tensor(logits), dim=-1)
#    return metric.compute(predictions=predictions, references=labels)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
    

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

# Train the model

train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model("./resnet_finetuned")
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
# Evaluate
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
print(metrics)

# Predict
#predictions = trainer.predict(dataset["test"])