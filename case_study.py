import torch
from dataset import NLIDataset, GPT3HallucinationDataset
from encoder import BaseSentenceEncoder, GPT2Encoder, FlanT5Encoder
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from discriminator import Discriminator
from tqdm import tqdm
import metric

DEVICE = "cpu"
ENCODER = FlanT5Encoder()
ENCODER_CKPT = "ckpt/FlanT5Encoder/2025-05-04_01:44:38/encoder-80.pt"
DISCRIMINATOR_CKPT = "ckpt/FlanT5Encoder/2025-05-04_01:44:38/discriminator-80.pt"


def main():
    encoder = ENCODER.to(DEVICE).eval()
    if ENCODER_CKPT is not None:
        encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    discriminator = Discriminator().to(DEVICE).eval()
    if DISCRIMINATOR_CKPT is not None:
        discriminator.load_state_dict(torch.load(
            DISCRIMINATOR_CKPT, map_location=DEVICE))

    premise = ["I am new here", "I have a class tonight", "I have a job","I have an easy job"]
    hypothesis = ["I am not familiar with here",
                  "I am free tonight", "I am free tonight","I am free tonight"]
    all_prompts = [f"premise: {x}. hypothesis: {y}" for x, y in zip(
        premise, hypothesis)]
    input_ids, attn_mask = encoder.tokenize(all_prompts)
    input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
    with torch.no_grad():
        word_emb, sentence_emb = encoder(input_ids, attn_mask)
        prob = discriminator(sentence_emb)
    print(prob)


if __name__ == "__main__":
    main()
