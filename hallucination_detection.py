import torch
from dataset import NLIDataset, GPT3HallucinationDataset
from encoder import BaseSentenceEncoder, GPT2Encoder, FlanT5Encoder
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from discriminator import Discriminator
from tqdm import tqdm
import metric

DEVICE = "cuda"
ENCODER = GPT2Encoder()
ENCODER_CKPT = "ckpt/GPT2Encoder/2025-05-04_02:27:25/encoder-120.pt"
DISCRIMINATOR_CKPT = "ckpt/GPT2Encoder/2025-05-04_02:27:25/discriminator-120.pt"
BATCH_SIZE = 64
USE_SPLIT_WIKI_TEXT = True


def main():
    dataset = GPT3HallucinationDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    encoder = ENCODER.to(DEVICE).eval()
    if ENCODER_CKPT is not None:
        encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    discriminator = Discriminator().to(DEVICE).eval()
    if DISCRIMINATOR_CKPT is not None:
        discriminator.load_state_dict(torch.load(
            DISCRIMINATOR_CKPT, map_location=DEVICE))

    # ground truth major_inaccurate, minor_inaccurate, accurate
    sample_count = [0, 0, 0]
    # true_major_inaccurate, true_minor_inaccurate, true_accurate
    true_pred = [0, 0, 0]
    # false_major_inaccurate, false_minor_inaccurate, false_accurate
    false_pred = [0, 0, 0]

    # count=-1
    for wiki_text, gpt3_sentence, label in tqdm(loader):
        # count+=1
        # if count<10:
        #     continue
        B = len(label)
        premise, hypothesis, index = [], [], []
        if USE_SPLIT_WIKI_TEXT:
            for i, (w, g) in enumerate(zip(wiki_text, gpt3_sentence)):
                wiki_text_split = w.split(".")
                premise += wiki_text_split
                hypothesis += [g]*len(wiki_text_split)
                index += [i]*len(wiki_text_split)
        else:
            premise = wiki_text
            hypothesis = gpt3_sentence
        all_prompts = [f"premise: {x}. hypothesis: {y}" for x, y in zip(
            premise, hypothesis)]  # [n_wiki_text_sentence (about 7B)]
        input_ids, attn_mask = encoder.tokenize(all_prompts)
        label = label.to(DEVICE)  # [B]
        index = torch.LongTensor(index).to(DEVICE)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
        with torch.no_grad():
            word_emb, sentence_emb = encoder(input_ids, attn_mask)
            prob = discriminator(sentence_emb)
        pred = prob.argmax(dim=-1)  # [n_wiki_text_sentence]

        if USE_SPLIT_WIKI_TEXT:
            grouped_max_pred = torch.zeros(B, dtype=torch.long).to(DEVICE)
            grouped_min_pred = torch.zeros(B, dtype=torch.long).to(DEVICE)
            grouped_max_pred.scatter_reduce_(-1, index,
                                             pred, reduce="amax", include_self=False)
            grouped_min_pred.scatter_reduce_(-1, index,
                                             pred, reduce="amin", include_self=False)
            grouped_pred = torch.ones_like(label)
            # if at least one of sentence in wiki_text identified as "accurate" together with gpt3_sentence
            # then that gpt3_sentence is regarded as accurate
            grouped_pred[grouped_min_pred == 0] = 0
            grouped_pred[grouped_max_pred == 2] = 2
            final_pred = grouped_pred
        else:
            final_pred = pred
        for i in range(3):
            sample_count[i] += (label == i).sum().item()
            true_pred[i] += ((label == i) &
                             (label == final_pred)).sum().item()
            false_pred[i] += ((label != i) & (final_pred == i)).sum().item()

    print(f"ckpt: {ENCODER_CKPT}, {DISCRIMINATOR_CKPT}")
    print(f"use_split_wiki_text: {USE_SPLIT_WIKI_TEXT}")
    print(f"sample_count: {sample_count}")
    print(f"pred_count: {[x+y for x, y in zip(true_pred, false_pred)]}")
    print(f"true_pred: {true_pred}")
    print(f"false_pred: {false_pred}")
    accuracy = metric.accuracy(sample_count, true_pred, false_pred)
    precision = metric.precision(sample_count, true_pred, false_pred)
    recall = metric.recall(sample_count, true_pred, false_pred)
    f1 = metric.f1_score(sample_count, true_pred, false_pred)
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")


if __name__ == "__main__":
    main()
