import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
import utils

log = utils.get_logger()

class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.best_dev_f1 = None
        self.best_dev_acc = None
        self.best_epoch_f1 = None
        self.best_epoch_acc = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_acc = ckpt["best_dev_acc"]
        self.best_epoch_acc = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):

        best_dev_f1, best_epoch_f1, best_state = self.best_dev_f1, self.best_epoch_f1, self.best_state
        best_dev_acc, best_epoch_acc = self.best_dev_acc, self.best_epoch_acc


        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            dev_acc, dev_f1, golds, preds = self.evaluate()
            log.info("[Dev set] [accuracy {:.4f}]".format(dev_acc))
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch_f1 = epoch
                log.info("Save the best f1 model.")
            if best_dev_acc is None or dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch_acc = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best acc model.")
            test_acc, test_f1, golds, preds = self.evaluate(test=True)
            log.info("[Test set] [accuracy {:.4f}]".format(test_acc))
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))

        # The best
        self.model.load_state_dict(best_state)
        log.info("Best in acc epoch {}:".format(best_epoch_acc))
        dev_accuracy, dev_f1, golds, preds = self.evaluate()
        log.info("[Dev set] [accuracy {:.4f}]".format(dev_accuracy))
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_accuracy, test_f1, golds, preds = self.evaluate(test=True)
        log.info("[Test set] [accuracy {:.4f}]".format(test_accuracy))
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))
        log.info("Best in f1 epoch {}:".format(best_epoch_f1))

        return best_dev_acc, best_epoch_acc, best_state, golds, preds

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        # ここからバッチごとの処理
        for step, batch in tqdm(enumerate(self.trainset),total=len(self.trainset), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            batch["encoded_context"] = batch["encoded_context"].to(self.args.device)
            batch["attention_mask"] = batch["attention_mask"].to(self.args.device)
            batch["turn_type_ids"] = batch["turn_type_ids"].to(self.args.device)
            batch["label"] = batch["label"].to(self.args.device)

            nll = self.model.get_loss(batch)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for step, batch in tqdm(enumerate(dataset),total=len(dataset), desc="test" if test else "dev"):
                golds.append(batch["label"])
                batch["encoded_context"] = batch["encoded_context"].to(self.args.device)
                batch["attention_mask"] = batch["attention_mask"].to(self.args.device)
                batch["turn_type_ids"] = batch["turn_type_ids"].to(self.args.device)
                batch["label"] = batch["label"].to(self.args.device)

                y_hat = self.model(batch)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            accuracy = metrics.accuracy_score(golds, preds)
            f1 = metrics.f1_score(golds, preds, average="macro")

        return accuracy, f1, golds, preds
