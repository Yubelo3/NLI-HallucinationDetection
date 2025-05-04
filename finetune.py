import torch
from dataset import NLIDataset
from encoder import BaseSentenceEncoder, GPT2Encoder, FlanT5Encoder
from logger import TBWriter
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from discriminator import BaseDiscriminator, L2Discriminator
from tqdm import tqdm


DEVICE = "cuda"
ENCODER = FlanT5Encoder()
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
    discriminator = L2Discriminator()
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
            logger.save_ckpt({"encoder": encoder}, epoch+1)


def train_epoch(
    encoder: BaseSentenceEncoder,
    discriminator: BaseDiscriminator,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
):
    encoder.train()
    sum_loss, sum_batches, sum_accuracy = 0.0, 0.0, 0.0
    sum_sim=0.0
    bar = tqdm(dataloader)
    for sentence1, sentence2, label in bar:
        B = label.shape[0]
        all_sentences = sentence1+sentence2
        input_ids, attn_mask = encoder.tokenize(all_sentences)
        label = label.to(DEVICE)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
        # sentence_emb: [2B x emb_dim]
        word_emb, sentence_emb = encoder(input_ids, attn_mask)
        sim = discriminator.get_similarity(
            sentence_emb[:B], sentence_emb[B:])  # [B]
        sim_loss=(sim.abs().sum()/B-0.666).abs()
        loss = (sim-label).abs().sum()/B+0.4*sim_loss
        sum_sim+=sim.abs().sum()/B

        bar.set_description(f"step_loss: {loss.item()}")
        sum_loss += loss.item()
        sum_batches += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mapped_label = mapped_output(label)
        mapped_pred = mapped_output(sim)
        accuracy = (mapped_label == mapped_pred).sum()/B
        sum_accuracy += accuracy
    print(f"avg sim: {sum_sim/sum_batches}")
    return sum_loss/sum_batches, sum_accuracy/sum_batches


def val_epoch(
    encoder: BaseSentenceEncoder,
    discriminator: BaseDiscriminator,
    dataloader: DataLoader,
):
    encoder.eval()
    sum_loss, sum_batches, sum_accuracy = 0.0, 0.0, 0.0
    bar = tqdm(dataloader)
    for sentence1, sentence2, label in bar:
        B = label.shape[0]
        all_sentences = sentence1+sentence2
        input_ids, attn_mask = encoder.tokenize(all_sentences)
        label = label.to(DEVICE)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
        # sentence_emb: [2B x emb_dim]
        with torch.no_grad():
            word_emb, sentence_emb = encoder(input_ids, attn_mask)
        sim = discriminator.get_similarity(
            sentence_emb[:B], sentence_emb[B:])  # [B]
        sim_loss=(sim.abs().sum()/B-0.666).abs()
        loss = (sim-label).abs().sum()/B+0.4*sim_loss
        bar.set_description(f"val_step_loss: {loss.item()}")
        sum_loss += loss.item()
        sum_batches += 1

        mapped_label = mapped_output(label)
        mapped_pred = mapped_output(sim)
        accuracy = (mapped_label == mapped_pred).sum()/B
        sum_accuracy += accuracy

    return sum_loss/sum_batches, sum_accuracy/sum_batches


def mapped_output(x: torch.Tensor, margin=0.333):
    mapped_label = x.detach().clone()
    mapped_label[mapped_label < -margin] = -1
    mapped_label[mapped_label > margin] = 1
    mapped_label[(mapped_label >= -margin) & (mapped_label <= margin)] = 0
    return mapped_label


if __name__ == "__main__":
    main()
