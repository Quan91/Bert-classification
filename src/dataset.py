import config
import torch


class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        # review = " ".join(review.split())

        encoding = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]

        return {
            "input_ids": input_ids.flatten(),  # torch.long
            "attention_mask": attention_mask.flatten(),  # 必须flatten()，不然报错
            "token_type_ids": token_type_ids.flatten(),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
        # return {
        #     "input_ids": torch.tensor(input_ids, dtype=torch.long),
        #     "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        #     "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        #     "targets": torch.tensor(self.target[item], dtype=torch.float)
        # }