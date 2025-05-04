import torch
import json
from torch.utils.data.dataset import Dataset, random_split


class NLIDataset(Dataset):
    # # ['annotator_labels', 'genre', 'gold_label', 'pairID', 'promptID', 'sentence1', 'sentence1_binary_parse', 'sentence1_parse', 'sentence2', 'sentence2_binary_parse', 'sentence2_parse']
    def __init__(
        self,
        filepath_matched: str = "data/dev_matched_sampled-1.jsonl",
        filepath_mismatched: str = "data/dev_mismatched_sampled-1.jsonl"
    ):
        # I want to use something that is similar to part of our course project (DPR)
        # labels matched to -1, 0, 1
        # contradiction => -1
        # neutral => 0
        # entailment => 1
        # then, similarity=dot(encode(sentence1),encode(sentence2))
        # loss=l2loss(similarity,label)
        # for samples that doesn't have "gold_label", take the average of annotator labels
        super().__init__()
        self.sentence1 = []
        self.sentence2 = []
        self.label = []
        self.n_matched_sample = 0
        self.n_mismatched_sample = 0
        def load_file(f) -> int:
            n_sample=0
            label_map = {"contradiction": -1.0, "neutral": 0.0, "entailment": 1.0}
            for line in f:
                data = json.loads(line)
                self.sentence1.append(data["sentence1"])
                self.sentence2.append(data["sentence2"])
                if data["gold_label"] in label_map:
                    self.label.append(label_map[data["gold_label"]])
                else:
                    annotator_labels=[label_map[x] for x in data["annotator_labels"]]
                    self.label.append(sum(annotator_labels)/len(annotator_labels))
                n_sample+=1
        with open(filepath_matched, "r", encoding="utf-8") as f:
            self.n_matched_sample=load_file(f)
        with open(filepath_mismatched, "r", encoding="utf-8") as f:
            self.n_mismatched_sample=load_file(f)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return [
            self.sentence1[index],
            self.sentence2[index],
            self.label[index]
        ]


class GPT3HallucinationDataset(Dataset):
    # ['gpt3_text', 'wiki_bio_text', 'gpt3_sentences', 'annotation', 'wiki_bio_test_idx', 'gpt3_text_samples']
    def __init__(
        self,
        filepath="data/wiki_bio_gpt3_hallucination.json"
    ) -> None:
        super().__init__()
        self.gpt3_sentence=[]
        self.wiki_text=[]
        self.label=[]
        with open(filepath, "r", encoding="utf-8") as f:
            data_list=json.load(f)
        label_map={"major_inaccurate":-1.0,"minor_inaccurate":0.0,"accurate":1.0}
        for data in data_list:
            wiki_text=data["wiki_bio_text"]
            gpt3_sentences=data["gpt3_sentences"]
            annotation=data["annotation"]
            for g,a in zip(gpt3_sentences,annotation):
                self.wiki_text.append(wiki_text)
                self.gpt3_sentence.append(g)
                self.label.append(label_map[a])
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return [
            self.wiki_text[index],
            self.gpt3_sentence[index],
            self.label[index]
        ]
    

if __name__ == "__main__":
    dataset = NLIDataset()
    train_size = int(0.7*len(dataset))
    val_size = int(0.2*len(dataset))
    test_size = len(dataset)-train_size-val_size
    trainset, valset, testset = random_split(
        dataset, [train_size, val_size, test_size])
    print(trainset[2])
    dataset=GPT3HallucinationDataset()
    print(dataset[2])