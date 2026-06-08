import re
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# load the data

dataset_stream = load_dataset(
    "Helsinki-NLP/europarl", "en-es", split="train", streaming=True
)
raw_data = list(dataset_stream.take(50000))


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


cleaned_pairs = []
for item in raw_data:
    en_txt = clean_text(item["translation"]["en"])
    es_txt = clean_text(item["translation"]["es"])
    if len(en_txt) < 2 or len(es_txt) < 2:
        continue
    if len(en_txt.split()) > 50 or len(es_txt.split()) > 50:
        continue
    cleaned_pairs.append({"en": en_txt, "es": es_txt})


# training tokenizer
all_text_iterator = [p["en"] for p in cleaned_pairs] + [p["es"] for p in cleaned_pairs]

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=12000, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

tokenizer.train_from_iterator(all_text_iterator, trainer=trainer)


# dataset class
class EnEsDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer

        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]

        en_ids = self.tokenizer.encode(pair["en"]).ids
        es_ids = self.tokenizer.encode(pair["es"]).ids

        encoder_input = en_ids + [self.eos_id]
        decoder_input = [self.bos_id] + es_ids
        decoder_target = es_ids + [self.eos_id]

        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_target, dtype=torch.long),
        )


random.seed(42)
random.shuffle(cleaned_pairs)

split = int(0.9 * len(cleaned_pairs))
train_data = EnEsDataset(cleaned_pairs[:split], tokenizer)
val_data = EnEsDataset(cleaned_pairs[split:], tokenizer)
