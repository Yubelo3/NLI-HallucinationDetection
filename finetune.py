import torch
from dataset import NLIDataset
from encoder import BaseSentenceEncoder, GPT2Encoder, FlanT5Encoder
from logger import TBWriter
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from discriminator import Discriminator
from tqdm import tqdm


DEVICE = "cuda"
ENCODER = GPT2Encoder()
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
BATCH_SIZE = 256
LR = 1e-4
SEED = 2025
EPOCHS = 200
SAVE_EVERY = 40


def main():
    logger = TBWriter(type(ENCODER).__name__)
    encoder = ENCODER.to(DEVICE)
    dataset = NLIDataset()
    discriminator = Discriminator().to(DEVICE)
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=LR, weight_decay=1e-4)
    train_size = int(TRAIN_RATIO*len(dataset))
    val_size = int(VAL_RATIO*len(dataset))
    test_size = len(dataset)-train_size-val_size
    trainset, valset, testset = random_split(
        dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE,
                            shuffle=False)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            encoder, discriminator, optimizer, train_loader)
        val_loss, val_acc = val_epoch(encoder, discriminator, val_loader)
        logger.add_scalar("train_loss", train_loss, epoch)
        logger.add_scalar("train_acc", train_acc, epoch)
        logger.add_scalar("val_loss", val_loss, epoch)
        logger.add_scalar("val_acc", val_acc, epoch)
        if (epoch+1) % SAVE_EVERY == 0:
            logger.save_ckpt(
                {"encoder": encoder, "discriminator": discriminator}, epoch+1)


def train_epoch(
    encoder: BaseSentenceEncoder,
    discriminator: Discriminator,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
):
    encoder.train()
    loss_func = torch.nn.CrossEntropyLoss()
    sum_loss, sum_batches, sum_accuracy = 0.0, 0.0, 0.0
    bar = tqdm(dataloader)
    for sentence1, sentence2, label in bar:
        B = label.shape[0]
        all_prompts = [f"premise: {x}. hypothesis: {y}" for x, y in zip(
            sentence1, sentence2)]
        input_ids, attn_mask = encoder.tokenize(all_prompts)
        label = label.to(DEVICE)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
        # sentence_emb: [B x emb_dim]
        word_emb, sentence_emb = encoder(input_ids, attn_mask)
        prob = discriminator(sentence_emb)
        pred = prob.argmax(dim=-1)
        accuracy = (pred == label).sum()/B
        sum_accuracy += accuracy
        loss = loss_func(prob, label)

        bar.set_description(f"step_loss: {loss.item()}")
        sum_loss += loss.item()
        sum_batches += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum_loss/sum_batches, sum_accuracy/sum_batches


def val_epoch(
    encoder: BaseSentenceEncoder,
    discriminator: Discriminator,
    dataloader: DataLoader,
):
    encoder.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    sum_loss, sum_batches, sum_accuracy = 0.0, 0.0, 0.0
    for sentence1, sentence2, label in dataloader:
        B = label.shape[0]
        all_prompts = [f"premise: {x}. hypothesis: {y}" for x, y in zip(
            sentence1, sentence2)]
        input_ids, attn_mask = encoder.tokenize(all_prompts)
        label = label.to(DEVICE)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
        # sentence_emb: [B x emb_dim]
        with torch.no_grad():
            word_emb, sentence_emb = encoder(input_ids, attn_mask)
            prob = discriminator(sentence_emb)
        pred = prob.argmax(dim=-1)
        accuracy = (pred == label).sum()/B
        sum_accuracy += accuracy
        loss = loss_func(prob, label)

        sum_loss += loss.item()
        sum_batches += 1
    return sum_loss/sum_batches, sum_accuracy/sum_batches


if __name__ == "__main__":
    main()
