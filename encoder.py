import torch
from transformers import GPT2Tokenizer, GPT2Model
from transformers import T5Tokenizer, T5EncoderModel
from typing import Tuple

class BaseSentenceEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def tokenize(self, sentences: str) -> Tuple[torch.LongTensor, torch.Tensor]:
        raise NotImplementedError()

    def forward(self, input_ids:torch.LongTensor, attention_mask:torch.Tensor):
        raise NotImplementedError()


class GPT2Encoder(BaseSentenceEncoder):
    def __init__(self, model="openai-community/gpt2") -> None:
        super().__init__()
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.model: GPT2Model = GPT2Model.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        for param in self.model.parameters():
            param.requires_grad = False
        # only train the last layer
        for param in self.model.h[-1].parameters():
            param.requires_grad = True

    def tokenize(self, sentences: str):
        tokenized = self.tokenizer(sentences, return_tensors="pt",padding=True)
        return tokenized["input_ids"], tokenized["attention_mask"]

    def forward(self, input_ids, attention_mask):
        word_emb: torch.Tensor = self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # [B x L x emb_dim]
        sentence_emb = (word_emb*attention_mask.unsqueeze(dim=-1)).sum(dim=-2)
        return word_emb, sentence_emb


class FlanT5Encoder(BaseSentenceEncoder):
    def __init__(self, model="google/flan-t5-base") -> None:
        super().__init__()
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model)
        self.model: T5EncoderModel = T5EncoderModel.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        for param in self.model.parameters():
            param.requires_grad = False
        # only train the last layer
        for param in self.model.encoder.block[-1].parameters():
            param.requires_grad = True

    def tokenize(self, sentences: str):
        tokenized = self.tokenizer(sentences, return_tensors="pt",padding=True)
        return tokenized["input_ids"], tokenized["attention_mask"]

    def forward(self, input_ids, attention_mask):
        word_emb: torch.Tensor = self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # [B x L x emb_dim]
        sentence_emb = (word_emb*attention_mask.unsqueeze(dim=-1)).sum(dim=-2)
        return word_emb, sentence_emb


if __name__ == "__main__":
    encoder = GPT2Encoder()
    input_ids, attn_mask = encoder.tokenize(["I am CSE student"])
    word_emb, sentence_emb = encoder(input_ids, attn_mask)
    print(word_emb.shape)
    print(sentence_emb.shape)
    encoder=FlanT5Encoder()
    input_ids, attn_mask = encoder.tokenize(["I am CSE student"])
    word_emb, sentence_emb = encoder(input_ids, attn_mask)
    print(word_emb.shape)
    print(sentence_emb.shape)
