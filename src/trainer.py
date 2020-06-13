import torch
from torch import nn
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import config
import tqdm

class BERTTrainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr, with_cuda):
        """
            :param model: BERT model which you want to train
            :param train_dataloader: train dataset data loader
            :param test_dataloader: test dataset data loader [can be None]
            :param lr: learning rate of optimizer
            :param betas: Adam optimizer betas
            :param weight_decay: Adam optimizer weight decay param
            :param with_cuda: traning with cuda
        """
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.model = model.to(self.device)
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [{"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],"weight_decay": 0.001,},
                                    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],"weight_decay": 0.0,},]

        total_steps = len(self.train_data) * config.EPOCHS
        self.optimizer = AdamW(optimizer_parameters, lr=config.LR)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        

    def train_fn(self, epoch, n_examples):
        model = self.model.train()
        losses = []
        correct_predictions = 0

        # data_iter = tqdm.tqdm(enumerate(self.train_data),
        #                       desc="EP_%s:%d" % ("train", epoch),
        #                       total=len(self.train_data),
        #                       bar_format="{l_bar}{r_bar}")

        for batch_idx, d in enumerate(self.train_data):
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            token_type_ids = d["token_type_ids"]
            targets = d["targets"]
            
            input_ids = input_ids.to(self.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
            attention_mask = attention_mask.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float).reshape(-1,1)
            
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            
            preds = torch.sigmoid(outputs) >= 0.5
            
            # accuracy = metrics.accuracy_score(targets, outputs)
            correct_predictions += torch.sum(preds == targets)

            loss = self.loss_fn(outputs, targets.view(-1, 1))
            loss = loss/config.ACCUMULATION
            
            loss.backward()
            
            if((batch_idx+1) % config.ACCUMULATION) == 0:
                losses.append(loss.item())
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
            if ((batch_idx+1) % config.ACCUMULATION) == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, batch_idx, len(self.train_data), 100. *
                    batch_idx/len(self.train_data), loss.item()
                ))                
        return correct_predictions.double() / n_examples, np.mean(losses)


    def eval_fn(self):
        model = self.model.eval()
        fin_targets = []
        fin_outputs = []
        
        with torch.no_grad():
            for batch_idx, d in enumerate(self.test_data):
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                token_type_ids = d["token_type_ids"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
        