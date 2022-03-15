import json
import torch, os
from torch.utils.data import Dataset, DataLoader
import utils

log = utils.get_logger()

def collator(minibatch_data):
    padding_value = 0
    batch_size = len(minibatch_data)

    data_to_return = {key: [] for key in minibatch_data[0].keys()}

    max_len = max([len(minibatch_data[i]["encoded_context"]) for i in range(batch_size)])

    for i in range(batch_size):
        for key in ["encoded_context", "turn_type_ids", "attention_mask"]:
            cur_len = len(minibatch_data[i][key])
            minibatch_data[i][key] += [padding_value for _ in range(max_len - cur_len)]

            data_to_return[key].append(minibatch_data[i][key])

        data_to_return["num_class"].append(minibatch_data[i]["num_class"])
        data_to_return["label"].append(minibatch_data[i]["label"])
        data_to_return["pair-id"].append(minibatch_data[i]["pair-id"])
        data_to_return["session-id"].append(minibatch_data[i]["session-id"])

    for key, value in data_to_return.items():
        data_to_return[key] = torch.Tensor(value).to(torch.long)

    return data_to_return

class ConversationRelDataModule():

    def __init__(self, num_class, dataset_path, tokenizer, batch_size, preprocessor, collator):
        self.num_class = num_class
        self.data_dir = dataset_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.collator = collator

        assert self.tokenizer is not None, "Must specify data tokenizer"
        assert self.collator is not None, "Must specify batch data collator"
        assert self.preprocessor is not None, "Must specify data pre-processor"

    def setup(self, stage):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = ConversationRelDataset(
                self.data_dir + "/train.txt",
                self.num_class,
                self.preprocessor,
                self.tokenizer
            )
            self.val_data = ConversationRelDataset(
                self.data_dir + "/dev.txt",
                self.num_class,
                self.preprocessor,
                self.tokenizer
            )

        # Assign test datasets for use in dataloader
        if stage == "test" or stage is None:
            self.test_data = ConversationRelDataset(
                self.data_dir + "/test.txt",
                self.num_class,
                self.preprocessor,
                self.tokenizer
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            # num_workers=10,
            batch_size=self.batch_size,
            collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            # num_workers=10,
            batch_size=self.batch_size,
            collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            # num_workers=10,
            batch_size=self.batch_size,
            collate_fn=self.collator)


class ConversationRelDataset(Dataset):
    def __init__(self, dataset_path, num_class, preprocess_func, tokenizer):

        self.dataset = []
        self.tokenizer = tokenizer

        four_class = {
            1: 0, 2: 0, 3: 0, 4: 0,
            5: 1, 6: 1,
            7: 2, 8: 2, 9: 2,
            10: 3, 11: 3, 12: 3, 13: 3}
        six_class = {
            1: 0, 2: 0,
            3: 1, 4: 1,
            5: 2, 6: 2,
            7: 3, 8: 3, 9: 3,
            10: 4,
            11: 5, 12: 5, 13: 5
        }

        assert num_class in [4, 6, 13], "only support 4, 6, 13 classes!"

        with open(dataset_path, "r") as f:
            for sample in f.readlines():
                sample = json.loads(sample)
                context = sample["context"]

                encoded_context = preprocess_func(context, self.tokenizer)
                self.dataset.append({
                    # "raw_context": context, # no need when training, just for debug
                    "pair-id": int(sample["pair-id"]),
                    "session-id": int(sample["session-id"]),
                    "num_class": num_class,
                    "encoded_context": encoded_context["input_ids"],
                    "turn_type_ids": encoded_context["turn_type_ids"],
                    "attention_mask": encoded_context["attention_mask"],
                })

                if num_class == 4:
                    self.dataset[-1]["label"] = four_class[int(sample["label"])]
                elif num_class == 6:
                    self.dataset[-1]["label"] = six_class[int(sample["label"])]
                else:
                    self.dataset[-1]["label"] = int(sample["label"]) - 1

        log.info("finished loading {} examples".format(len(self.dataset)))
        # print("- finished loading {} examples".format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class ConversationRelPreprocessor():

    # Convert a string in a sequence of ids, using the tokenizer
    def bert_preprocess(context, tokenizer):

        context = [tokenizer.encode(c) for c in context]

        # input_ids = [101, utterance1 ids, 102, utterance2 ids, 102, ・・・, 102]
        input_ids = [101] + [id for c in context for id in c[1:]]
        turn_type_ids = [1 for _ in range(len(input_ids))]
        attn_mask = [1 for _ in range(len(input_ids))]

        original_len = len(input_ids)
        input_ids = input_ids[: min(512, original_len)]
        turn_type_ids = turn_type_ids[: min(512, original_len)]
        attn_mask = attn_mask[: min(512, original_len)]

        return {"input_ids": input_ids, "turn_type_ids": turn_type_ids, "attention_mask": attn_mask}
