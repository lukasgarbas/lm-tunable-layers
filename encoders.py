import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import logging
from typing import Union


class LanguageModel(torch.nn.Module):
    def __init__(
        self,
        model: str = "bert-base-cased",
        hidden_dropout_prob: float = 0.1,
        tunable_layers: Union[str, int] = "all",
    ):
        """
        Downloads any pretrained transformer-based language model from Huggingface and
        uses CLS token as sentence or sentence pair embedding
        """

        super().__init__()

        # GPU or CPU device
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # load auto tokenizer from huggingface transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # ignore warning messages: we will use models from huggingface model hub checkpoints
        # some layers won't be used (i.e. pooling layer) and this will throw warning messages 
        logging.set_verbosity_error()

        # load transformer config file and set different dropout if needed
        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        print(self.config)
        self.config.hidden_dropout_prob=hidden_dropout_prob

        # load the model using auto class
        self.model = AutoModel.from_pretrained(model, config=self.config)

        # name of the language model
        self.name = str(model).rsplit('/')[-1]

        # sets how many layers to tune
        self.tunable_layers = tunable_layers

        # BERT, SpanBERT, ELECTRA, ERNIE models have module named 'encoder.layer' with 11 or 12 layers
        self.layers = [module for module in self.model.encoder.layer.children()]

        # when initializing, models are in eval mode by default
        self.model.eval()
        self.model.to(self.device)

        # hidden size of the original model
        self.embedding_length = self.config.hidden_size

        # check whether CLS is at beginning or end
        # BERT, SpanBERT, ERNIE, ELECTRA use cls at the start
        tokens = self.tokenizer.encode('a')
        self.cls_position = 0 if tokens[0] == self.tokenizer.cls_token_id else -1


    def forward(self, sentences):
        # sentences can be tuples for sentence-pair tasks
        if isinstance(sentences[0], tuple) and len(sentences) == 2:
            sentences = list(zip(sentences[0], sentences[1]))

        # tokenize sentences and sentence pairs
        # huggingface tokenizers already support batch tokenization
        tokenized_sentences = self.tokenizer(sentences,
                                            padding=True,
                                            truncation=True,
                                            return_tensors='pt')

        # tokenizer returns input ids: unique ids for subwords
        input_ids = tokenized_sentences["input_ids"].to(self.device)

        # and attention masks: 1's for for subwords, 0's for padding tokens
        mask = tokenized_sentences["attention_mask"].to(self.device)

        # some models use token type ids for sentence-pair tasks (i.e. next sentence prediction)
        use_token_type = True if "token_type_ids" in tokenized_sentences else False
        if use_token_type:
            token_type_ids = tokenized_sentences["token_type_ids"].to(self.device)

        for param in self.model.parameters():
            if self.tunable_layers == "all":
                # enable gradients for all LM parameters
                param.requires_grad = True
            else:
                # disable gradients for all LM parameters
                param.requires_grad = False

        # enable gradients just for selected top layers
        if type(self.tunable_layers) == int:
            for i in range(self.tunable_layers):
                for param in self.layers[-1*(i+1)].parameters():
                    param.requires_grad = True

        if use_token_type:
            subword_embeddings = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=mask,
            ).last_hidden_state
        else:
            subword_embeddings = self.model(
                input_ids=input_ids,
                attention_mask=mask,
            ).last_hidden_state

        # use cls token for sentence-level tasks
        cls_embedding  = torch.stack(
            [embedding[self.cls_position] for embedding in subword_embeddings]
        )

        return cls_embedding
