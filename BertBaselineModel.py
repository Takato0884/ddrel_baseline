import torch, json, math
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
from collections import OrderedDict
from argparse import ArgumentParser
from torch.optim import Adam, SGD, Adadelta
import utils

log = utils.get_logger()

class BertBaselineModel(nn.Module):
    def __init__(self, input_dim, hidden_size, args):
        super(BertBaselineModel, self).__init__()

        num_class = args.num_class

        # Model Architecture
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, num_class)

        if args.class_weight == "True":
            log.info("loss_weights")
            # loss_weightsはそれぞれのラベルの比率
            if num_class == 13:
                self.loss_weights = torch.tensor([15.03283582089552, 77.47692307692307, 29.27906976744186, \
                11.367945823927766, 3.3889636608344547, 43.41379310344828, 5.959763313609468, \
                387.3846153846154, 62.95, 11.657407407407407, 8.88183421516755, 31.873417721518987, \
                15.543209876543212]).to(args.device)
            elif num_class == 6:
                self.loss_weights = torch.tensor([12.59, 8.188617886178863, 3.143570536828964, 5.368869936034114, \
                11.657407407407407, 4.800762631077217]).to(args.device)
            elif num_class == 4:
                self.loss_weights = torch.tensor([4.961576354679803, 3.143570536828964, 5.368869936034114, \
                3.400405131667792]).to(args.device)

            self.nll_loss = nn.NLLLoss(self.loss_weights)

        elif args.class_weight == "False":
            log.info("Not loss_weights")
            self.nll_loss = nn.NLLLoss()

    def get_prob(self, context_data):

        output = self.bert(
            input_ids=context_data["encoded_context"],
            attention_mask=context_data["attention_mask"],
            token_type_ids=context_data["turn_type_ids"],
        )

        # Use [CLS] pooler_output for prediction.
        hidden1 = self.drop(F.relu(self.lin1(output["pooler_output"])))
        scores = self.lin2(hidden1)
        log_prob = F.log_softmax(scores, dim=-1)

        return log_prob

    def forward(self, context_data):

        log_prob = self.get_prob(context_data)
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat

    def get_loss(self, context_data):

        log_prob = self.get_prob(context_data)
        loss = self.nll_loss(log_prob, context_data["label"])

        return loss
