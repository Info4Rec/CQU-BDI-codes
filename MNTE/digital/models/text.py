import torch
from torch import nn as nn

from transformers import BertTokenizer, BertModel, BertConfig


def EncoderText(opt):
    print('Using BERT text encoder')
    model = EncoderTextBERT(opt)
    return model

class EncoderTextBERT(nn.Module):
    def __init__(self, opt, order_embeddings=False, mean=True, post_transformer_layers=0):
        super().__init__()
        bert_config = BertConfig.from_pretrained(opt.bert_config,
                                                 output_hidden_states=True,
                                                 num_hidden_layers=opt.text_extraction_hidden_layer)
        bert_model = BertModel.from_pretrained(opt.bert_model, config=bert_config)
        self.order_embeddings = order_embeddings
        self.vocab_size = bert_model.config.vocab_size
        self.hidden_layer = opt.text_extraction_hidden_layer
        self.tokenizer = BertTokenizer.from_pretrained(opt.bert_text)
        self.bert_model = bert_model
        self.word_embeddings = self.bert_model.get_input_embeddings()
        self.post_transformer_layers = post_transformer_layers
        self.map = nn.Linear(opt.text_word_dim, opt.embed_size)
        self.mean = mean

    def forward(self, x, lengths):
        '''
        x: tensor of indexes (LongTensor) obtained with tokenizer.encode() of size B x ?
        lengths: tensor of lengths (LongTensor) of size B
        '''
        max_len = max(lengths)
        attention_mask = torch.ones(x.shape[0], max_len)
        for e, l in zip(attention_mask, lengths):
            e[l:] = 0
        attention_mask = attention_mask.to(x.device)


        outputs = self.bert_model(x, attention_mask=attention_mask)
        outputs = outputs[2][-1]

        if self.mean:
            x = outputs.mean(dim=1)
        else:
            x = outputs[:, 0, :]     # from the last layer take only the first word

        out = self.map(x)

        # normalization in the joint embedding space
        # out = l2norm(out)

        # take absolute value, used by order embeddings

        return out, outputs

    def get_finetuning_params(self):
        return list(self.bert_model.parameters())