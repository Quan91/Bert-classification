import config
import dataset
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from trainer import BERTTrainer


def run():
    print('1.Loading data...')
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    
    # only train 2000 entries
    dfx = dfx[:2000]
    dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    print('Creating dataset...')
    train_dataset = dataset.BERTDataset(
        review=df_train.review.values, target=df_train.sentiment.values
    )
    
    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values, target=df_valid.sentiment.values
    )
    
    print('Creating dataloader...')
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    print('Building Bert Model...')
    model = BERTBaseUncased()
    
    print("Creating BERT Trainer...")
    trainer = BERTTrainer(model=model,
                          train_dataloader=train_data_loader, 
                          test_dataloader=valid_data_loader, 
                          lr=config.LR, 
                          with_cuda=config.USE_CUDA)

    
    # model = nn.DataParallel(model)

    print('Training Start...')
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        train_acc, train_loss = trainer.train_fn(epoch, len(df_train))
        print(f'Train loss: {train_loss} Train accuracy: {train_acc:.4%}')
        
        outputs, targets = trainer.eval_fn()
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy:.2%}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()