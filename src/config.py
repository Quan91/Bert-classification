import transformers


MAX_LEN = 400
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 2
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../input/model.bin"
TRAINING_FILE = "../input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, 
                                                       do_lower_case=True)
ACCUMULATION = 8
LR = 3e-5