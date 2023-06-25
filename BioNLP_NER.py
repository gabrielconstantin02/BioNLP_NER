from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
from tqdm import tqdm

import re
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, classification_report
import numpy as np
import os
import argparse

import random

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
from time import perf_counter

import torch
import torch.nn as nn

# import jax
# jax.random.PRNGKey(seed)

from unidecode import unidecode

from torch.utils.data import Dataset, DataLoader

import json
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


from datasets import load_dataset
dataset = load_dataset("species_800")


train_dataset = dataset["train"]
valid_dataset = dataset["validation"]
test_dataset = dataset["test"]

# counter = 0
# for item in dataset["train"]:
#     counter += len(item['tokens'])
# counter /= len(dataset["train"]) # 26 wp
# import pdb; pdb.set_trace();

# print(dataset)
# print(dataset["train"])
# print(len(dataset["train"]))
print(f"Train dataset contains {len(train_dataset)} instances.")  
print(f"Validation dataset contains {len(valid_dataset)} instances.")  
print(f"Test dataset contains {len(test_dataset)} instances.")

class MyModel():
  def __init__(self, opt):
    # do here any initializations you require
    # addings seeds
    seed = opt.seed 
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)

    # Defaults
    self.EPOCHS = opt.epochs
    self.BATCH_SIZE = opt.batch_size
    self.NUM_LAYERS_FROZEN = opt.freeze

    self.NUM_CLASSES = 3
    self.MAX_LENGTH = 128
    self.LEARNING_RATE = 1e-4
    # self.LEARNING_RATE = 5e-5

    self.classes = [ "0", "1", "2"]
    self.opt = opt

    # train_data, train_labels = self.read_dataset(train_dataset, tokenizer=tokenizer)
    # self.model = AutoModelForTokenClassification.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1", num_labels=self.NUM_CLASSES)
    # self.tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

    # self.model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=self.NUM_CLASSES)
    # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    self.model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=self.NUM_CLASSES)
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # self.model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=self.NUM_CLASSES)
    # self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

    # self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=self.NUM_CLASSES, ignore_mismatched_sizes=True)
    # self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    
    # self.model = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-base", num_labels=self.NUM_CLASSES, ignore_mismatched_sizes=True)
    # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", add_prefix_space=True)

    # self.model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2", num_labels=self.NUM_CLASSES, ignore_mismatched_sizes=True)
    # self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    # self.model = AutoModelForTokenClassification.from_pretrained("nlpie/distil-biobert", num_labels=self.NUM_CLASSES, ignore_mismatched_sizes=True)
    # self.tokenizer = AutoTokenizer.from_pretrained("nlpie/distil-biobert")

 
    # for param in self.model.bert.parameters():
    #   param.requires_grad = False

    # for param in self.model.bert.embeddings.parameters():
    #   param.requires_grad = False
    # for layer in self.model.bert.encoder.layer[:self.NUM_LAYERS_FROZEN]:

    # for param in self.model.roberta.embeddings.parameters():
    #   param.requires_grad = False
    # for layer in self.model.roberta.encoder.layer[:self.NUM_LAYERS_FROZEN]:
    # for param in self.model.deberta.embeddings.parameters():
    #   param.requires_grad = False
    # for layer in self.model.deberta.encoder.layer[:self.NUM_LAYERS_FROZEN]:
    for param in self.model.bert.embeddings.parameters():
      param.requires_grad = False
    for layer in self.model.bert.encoder.layer[:self.NUM_LAYERS_FROZEN]:
        for param in layer.parameters():
            param.requires_grad = False

  def load(self):
    self.model.load_state_dict(torch.load(self.opt.weights))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)

  def train(self, train_data, validation_data):
    # Prepairing the data
    print("Preparing data")
    X_train, y_train = self.read_dataset(train_data, tokenizer=self.tokenizer)
    X_val, y_val = self.read_dataset(validation_data, tokenizer=self.tokenizer)

    # Computing weights for our evaluation
    proper_labels = []
    for sequence in y_train:
        for label in sequence:
            if label != -100:
                proper_labels.append(int(label))
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(0, self.NUM_CLASSES), y=proper_labels)
    self.class_weights = {idx:weight for idx, weight in enumerate(weights)}

    weights_base = np.zeros(3)
    for sequence in y_train:
        for label in sequence:
            if label != -100:
                weights_base[label] += 1
    # max_weight = np.max(weights_base)
    max_weight = np.sum(weights_base)
    for index, weight in enumerate(weights_base):
        weights_base[index] = max_weight / weights_base[index] 
    # import pdb; pdb.set_trace();

    self.weights = weights
    self.weights_base = weights_base

    # Creating the datasets
    train_dataset = MyModel.MyDataset(X_train, y_train)
    validation_dataset = MyModel.MyDataset(X_val, y_val)

    # Creating the dataloaders
    train_dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=self.BATCH_SIZE,
      shuffle=True
    )
    validation_dataloader = DataLoader(
      dataset=validation_dataset,
      batch_size=self.BATCH_SIZE
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move the model to GPU (when available)
    self.model.to(device)

    # create a SGD optimizer
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    # set up loss function
    loss_criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float).to(device))
    print("Training:")
    train_losses = []
    train_accuracies = []
    train_f1s = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    best_acc = -1
    for epoch in range(1, self.EPOCHS + 1):
        train_loss, train_accuracy, train_f1, _ = self.train_epoch(self.model, train_dataloader, loss_criterion, optimizer, device)
        val_loss, val_accuracy, val_f1, val_labels, val_predictions = self.eval_epoch(self.model, validation_dataloader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        print('\nEpoch %d'%(epoch))
        print('train loss: %10.8f, accuracy: %10.8f, f1: %10.8f\n'%(train_loss, train_accuracy, train_f1))
        print('val loss: %10.8f, accuracy: %10.8f, f1:%10.8f\n'%(val_loss, val_accuracy, val_f1))

        if best_acc < val_f1:
          best_acc = val_f1
          torch.save(self.model.state_dict(), os.path.join(self.opt.model_output_path, f"best.pt"))

        # sample_weights = compute_sample_weight(self.class_weights, val_labels)
        # print(sample_weights)
        # print(classification_report(val_labels, val_predictions, labels=np.unique(np.array(val_labels))))
        # print(classification_report(val_labels, val_predictions, labels=np.unique(np.array(val_labels)), sample_weight=sample_weights))
        cf_matrix = confusion_matrix(val_labels, val_predictions, labels=np.unique(np.array(val_labels)))
        sns.set(font_scale=0.7)
        sns.set(rc={'figure.figsize':(25,25)})
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d', norm=LogNorm())

        ax.set_title('Confusion Matrix on logarithmic scale\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        plt.savefig(os.path.join(self.opt.project_name, f'confusion_matrix_log_{epoch}.png'), dpi=150)
        plt.close()

        torch.save(self.model.state_dict(), os.path.join(self.opt.model_output_path, f"last.pt"))

        self.plot_gen(train_losses, val_losses, "loss") 
        self.plot_gen(train_accuracies, val_accuracies, "accuracy")
        self.plot_gen(train_f1s, val_f1s, "F1")


  def predict(self, test_data_file):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      X_test, y_test, item_lengths = self.read_dataset(test_data_file, tokenizer=self.tokenizer, return_lengths=True)
      test_dataset = MyModel.MyDataset(X_test, y_test)

      test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=self.BATCH_SIZE
      )
      # all_predictions, all_labels = self.test_epoch(self.model, test_dataloader, device)
      test_loss, test_accuracy, test_f1, test_labels, test_predictions = self.eval_epoch(self.model, test_dataloader, device)


      print('Test loss: %10.8f, accuracy: %10.8f, f1:%10.8f\n'%(test_loss, test_accuracy, test_f1))

      cf_matrix = confusion_matrix(test_labels, test_predictions, labels=np.unique(np.array(test_labels)))
      sns.set(font_scale=0.7)
      sns.set(rc={'figure.figsize':(25,25)})
      ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d', norm=LogNorm())

      ax.set_title('Confusion Matrix on logarithmic scale\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values ')
      plt.savefig(os.path.join(self.opt.project_name, f'test_confusion_matrix.png'), dpi=150)
      plt.close()

      return test_f1

  def train_epoch(self, model, train_dataloader, loss_crt, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_f1 = 0.0
    num_batches = len(train_dataloader)
    predictions = []
    labels = []
    for idx, batch in tqdm(enumerate(train_dataloader)):
        batch_data, batch_labels = batch
        sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
        sequence_masks = batch_data['attention_mask'].to(device)
        batch_labels = batch_labels.to(device)

        raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks, labels=batch_labels)
        loss, output = raw_output['loss'], raw_output['logits']
        logits = output.view(-1, model.num_labels)
        batch_predictions = torch.argmax(logits, dim=1)

        proper_labels = batch_labels.view(-1) != -100
        loss = loss_crt(logits, batch_labels.view(-1))

        filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
        filtered_predictions = torch.masked_select(batch_predictions, proper_labels)

        labels += filtered_labels.squeeze().tolist()
        predictions += filtered_predictions.tolist()

        #sample_weights = [self.weights[label] for label in filtered_labels.cpu().numpy()]
        sample_weights = compute_sample_weight(self.class_weights, filtered_labels.squeeze().tolist())
        batch_acc = balanced_accuracy_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy(), sample_weight=sample_weights)
        epoch_acc += batch_acc
        batch_f1 = f1_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy(), average='macro')#, sample_weight=sample_weights)
        epoch_f1 += batch_f1

        loss_scalar = loss.item()

        # if idx % 500 == 0:
        #     print(epoch_acc/(idx + 1))
        #     print(batch_predictions)

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=10
        )

        model.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss_scalar

    epoch_loss = epoch_loss / num_batches
    epoch_acc = epoch_acc / num_batches
    epoch_f1 = epoch_f1 / num_batches

    return epoch_loss, epoch_acc, epoch_f1, labels
  
  def eval_epoch(self, model, val_dataloader, device):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_f1 = 0.0
    num_batches = len(val_dataloader)
    predictions = []
    labels = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader)):
            batch_data, batch_labels = batch
            sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
            sequence_masks = batch_data['attention_mask'].to(device)
            batch_labels = batch_labels.to(device)

            raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks, labels=batch_labels)
            loss, output = raw_output['loss'], raw_output['logits']
            logits = output.view(-1, model.num_labels)
            batch_predictions = torch.argmax(logits, dim=1)

            proper_labels = batch_labels.view(-1) != -100

            filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
            filtered_predictions = torch.masked_select(batch_predictions, proper_labels)

            labels += filtered_labels.squeeze().tolist()
            predictions += filtered_predictions.tolist()

            sample_weights = [self.weights[label] for label in filtered_labels.cpu().numpy()]
            batch_acc = balanced_accuracy_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy(), sample_weight=sample_weights)
            epoch_acc += batch_acc

            batch_f1 = f1_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy(), average='macro')
            epoch_f1 += batch_f1

            loss_scalar = loss.item()

            epoch_loss += loss_scalar

    epoch_loss = epoch_loss / num_batches
    epoch_acc = epoch_acc / num_batches
    epoch_f1 = epoch_f1 / num_batches

    return epoch_loss, epoch_acc, epoch_f1, labels, predictions

  def test_epoch(self, model, test_dataloader, device):
    model.eval()
    epoch_loss = 0.0
    num_batches = len(test_dataloader)
    predictions = []
    labels = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader)):
            batch_data, batch_labels = batch
            sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
            sequence_masks = batch_data['attention_mask'].to(device)
            batch_labels = batch_labels.to(device)

            offset_mapping = batch_data['offset_mapping']

            raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks)
            output =  raw_output['logits']
            logits = output.view(-1, model.num_labels)
            batch_predictions = torch.argmax(logits, dim=1)
            

            filtered_predictions = []
            proper_labels = batch_labels.view(-1) != -100
            filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
            # print(f'len(filtered_labels): {len(filtered_labels)}')
            labels += filtered_labels.squeeze().tolist()

            for index, offset in enumerate(offset_mapping.view(-1, 2)):
                if offset[0] == 0 and offset[1] != 0:
                    filtered_predictions.append(batch_predictions[index].item())
            # print(f'len(filtered_predictions): {len(filtered_predictions)}')
            predictions += filtered_predictions

    return predictions, labels
  
  def get_tokens(self, dataset):
    token_list = ['0' for i in range(self.NUM_CLASSES)]
    for item in dataset:
        for id_ner_id in range(len(item['ner_ids'])):
            token_list[int(item['ner_ids'][id_ner_id])] = item['ner_tags'][id_ner_id]
    return token_list

  def read_dataset(self, dataset, tokenizer, train=True, return_lengths=False):
    data = []
    labels = []
    max_length = 0
    reshaped_dataset = []
    reshaped_labels = []
    item_lengths = []
    reshaped_length = 110
    for item in dataset:
        prelucrate_item = []
        item_lengths.append(len(item['ner_tags']))

        for token in item['tokens']:
            prelucrate_item.append(re.sub(r"\W+", 'n', token))

        for i in range(0, len(prelucrate_item), reshaped_length):
            reshaped_dataset.append(prelucrate_item[i: min(i + reshaped_length, len(prelucrate_item))])
            # print(item.keys())
            reshaped_labels.append( item['ner_tags'][i: min(i + reshaped_length, len(item['ner_tags']))])
            
    for index in range(len(reshaped_dataset)):
        items, sequence_labels =  reshaped_dataset[index], reshaped_labels[index]
        sequence = tokenizer(
            items,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_offsets_mapping=True

        )
        sequence = {key: torch.as_tensor(value) for key, value in sequence.items()}
        data.append(sequence)

        if train:
            encoded_labels = np.ones(len(sequence["offset_mapping"]), dtype=int) * -100
            # set only labels whose first offset position is 0 and the second is not 0
            i = 0
            for idx, offsets in enumerate(sequence["offset_mapping"]):
                if offsets[0] == 0 and offsets[1] != 0:
                    # overwrite label
                    encoded_labels[idx] = sequence_labels[i]
                    i += 1

            # max_length = max(len(sequence), max_length)
            labels.append(torch.as_tensor(encoded_labels))
    # print(max_length)
    if train:
        if return_lengths:
            return data, labels, item_lengths
        return data, labels

    return data

  def plot_gen(self, train_data, val_data, metric, x_label="epochs"):
    plt.plot(range(0, len(train_data)), train_data, 'r', label='Training ' + metric)
    plt.plot(range(0, len(train_data)), val_data, 'b', label='Validation ' + metric)
    plt.title("Training and Validation " + metric)
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(os.path.join(self.opt.project_name, metric + '.png'))
    #plt.show()
    plt.close()

  class MyDataset(Dataset):
      def __init__(self, data, labels):
          super().__init__()
          self.data = data
          self.labels = labels

      def __getitem__(self, index):
          return self.data[index], self.labels[index]

      def __len__(self):
          return len(self.labels)

  class TestDataset(Dataset):
      def __init__(self, data, labels):
          super().__init__()
          self.data = data

      def __getitem__(self, index):
          return self.data[index]

      def __len__(self):
          return len(self.data)

def parse_opt(known=False): 
  parser = argparse.ArgumentParser()
  parser.add_argument('--project-name', default='runs/train/train_test', help='save to project/name')
  parser.add_argument('--weights', type=str, default='best.pt', help='weights path to use on test')
  parser.add_argument('--epochs', type=int, default=15, help='total training epochs')
  parser.add_argument('--batch-size', type=int, default=64, help='total batch size')
  parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Whether to only test or run train as well')
  parser.add_argument('--freeze', type=int, default=6, help='Number of frozen layers')
  parser.add_argument('--seed', type=int, default=8, help='Global training seed')


  return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == "__main__":
  # parser stuffs
  opt = parse_opt()
  opt.model_output_path = os.path.join(opt.project_name, 'weights')
  if opt.mode == 'train':
    os.makedirs(opt.project_name, exist_ok=True)
    os.makedirs(opt.model_output_path, exist_ok=True)

  if opt.mode == "train":
    model = MyModel(opt)
    model.train(train_dataset, valid_dataset)

    # inference
    start_time = perf_counter()
    f1_strict_score = model.predict(test_dataset)
    stop_time = perf_counter()
  else:
    model = MyModel(opt)
    model.load(opt.weights)

    # inference
    start_time = perf_counter()
    f1_strict_score = model.predict(test_dataset)
    stop_time = perf_counter()


  print(f"Predicted in {stop_time-start_time:.2f}.")
  print(f"F1-strict score = {f1_strict_score:.5f}") # this is the score we want :
