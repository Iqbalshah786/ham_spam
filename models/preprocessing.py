import torch
from transformers import BertTokenizer, BertModel

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-german-cased')
        self.max_length = 512

    def preprocess(self, text):
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokens

    def get_embeddings(self, text):
        tokens = self.preprocess(text)
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
        # Use the [CLS] token's embedding as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
