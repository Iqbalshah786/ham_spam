#models / preprocessing.py
import torch
from transformers import BertTokenizer, BertModel

class TextPreprocessor:
    def __init__(self, max_length: int = 512):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-german-cased')
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = self.bert_model.to(self.device)  # Move model to device

    def preprocess(self, text: str):
        """
        Tokenize input text for BERT.
        Args:
            text (str): The input text to tokenize.
        Returns:
            dict: Tokenized text as tensors.
        """
        tokens = self.tokenizer(
            text,
            padding='max_length',  # Fixed padding to max_length for now
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokens

    def get_embeddings(self, text: str):
        """
        Generate sentence embedding using BERT.
        Args:
            text (str): The input text to process.
        Returns:
            torch.Tensor: The [CLS] token embedding for the text.
        """
        tokens = self.preprocess(text)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}  # Move tokens to device

        with torch.no_grad():
            outputs = self.bert_model(**tokens)
        
        # Use the [CLS] token's embedding as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def batch_preprocess(self, texts: list[str]):
        """
        Preprocess a batch of texts for tokenization and embedding.
        Args:
            texts (list[str]): List of texts to process.
        Returns:
            dict: Tokenized batch of texts.
        """
        tokens = self.tokenizer(
            texts,
            padding=True,  # Pad to the length of the longest sentence in the batch
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: value.to(self.device) for key, value in tokens.items()}

    def batch_get_embeddings(self, texts: list[str]):
        """
        Generate embeddings for a batch of texts.
        Args:
            texts (list[str]): List of texts to process.
        Returns:
            torch.Tensor: Batch of [CLS] token embeddings.
        """
        tokens = self.batch_preprocess(texts)

        with torch.no_grad():
            outputs = self.bert_model(**tokens)
        
        embeddings = outputs.last_hidden_state[:, 0, :]  # Batch of [CLS] embeddings
        return embeddings
