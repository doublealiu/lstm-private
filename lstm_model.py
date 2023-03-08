import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as torchf
from torch.distributions import Categorical

from vocab import Vocabulary


# The baseline model
# Resnet50 for encoding
# LSTM for decoding
# Writeup recommends:
# 2 layers
# 512 hidden units
class LSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab: Vocabulary, num_layers=2):
        super(LSTMModel, self).__init__()
        self.model_type = "LSTM2"
        self.num_layers = num_layers
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): # freeze params
            param.requires_grad = False
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnetfc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.word_embedder = nn.Embedding(self.vocab_size, embedding_size)
        self.decoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # if batch first set to false it will accept
        # and output tensors w/ size (L, N, I) where
        # L is size of sequence
        # N is batch size
        # I is size of each vector in sequence
        # if batch_first=True then tensors will be
        # of the form (N, L, I), which is just nicer
        # to deal w/
        # for feeding hiddens into
        self.hidden2output = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, images, captions):
        """
        Forward pass. nuff' said.
        Params:
            images: 2d tensor w/ shape (N, img_size)
            captions: 32 tensor w/ shape (N, seq len)
                A batch of target captions for the corresponding
                image. Each sequence should be a list of integer indices
                obtained by running through tokenized caption
                through vocabulary object. padding where appropriate
        Outputs:
            3d tensor w/ shape (N, vocab size, seq len)
                A tensor containing the one-hot encoded word
                predictions. This specific ordering used so
                that it can be fed into the loss function.
        """
        # encode images to LSTM hidden state
        batch_size = images.size()[0]
        # hidden_states = self.resnetfc(self.resnet(images).squeeze_()) # (N, num_layers * hidden_size)
        # hidden_states = torch.reshape(hidden_states, (batch_size, self.num_layers, self.hidden_size))
        # hidden_states = torch.permute(hidden_states, (1, 0, 2)) # (num_layers, batch_size, hidden_size)
        features = self.resnetfc(self.resnet(images).squeeze_()) # (N, embedding_size)
        features = features.unsqueeze(dim=1)
        embedded = self.word_embedder(captions[:, :-1]) # (N, seq len, embedding_size)
        embedded = torch.cat((features, embedded), dim=1)
        # print(f'embedded: {embedded.size()}')
        # decode w/ lstm
        # (N, seq len, hidden_size)
        final_hiddens, (_, _) = self.decoder(embedded)
        # print(f'final_hiddens: {final_hiddens.size()}')
        output = self.hidden2output(final_hiddens) # (N, seq len, vocab size)
        return output.permute(0, 2, 1) # (N, vocab size, seq len)

    def generate(self, images, config):
        """
        Generate captions for the images in the batch
        based completely off hidden states encoded by
        the model.
        Params:
            images: 2d tensor w/ shape (N, img_size)
            config: The generation config. See "generation"
                in default.json for more details
        Output:

        """
        # read config stuff.
        max_len = config['max_length']
        deterministic = config['deterministic']
        temperature = config['temperature']

        # encode images to LSTM hidden state
        batch_size = images.size()[0]
        # hidden_states = self.resnetfc(self.resnet(images).squeeze_()) # (N, num_layers * hidden_size)
        # hidden_states = torch.reshape(hidden_states, (batch_size, self.num_layers, self.hidden_size))
        # hidden_states = torch.permute(hidden_states, (1, 0, 2)) # (num_layers, batch_size, hidden_size)
        # cell_states = hidden_states
        hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_size).detach()
        cell_states = torch.zeros(self.num_layers, batch_size, self.hidden_size).detach()

        # we want to convert or image encodings into captions
        # generated by the model. We will keep feeding outputs to
        # the model until it hits <end>, in which case we'll
        # stop for that specific batch example
        captions = [["<pad>"] * (max_len + 1) for i in range(batch_size)]
        # start off w/ <start>
        for i in range(batch_size):
            captions[i][0] = "<start>"
        keep_generating = [True] * batch_size
        iter = 0
        num_complete = 0
        while iter < max_len + 1 and num_complete < batch_size:
            # encode most recent word using vocab
            if iter == 0:
                features = self.resnetfc(self.resnet(images).squeeze_())  # (N, embedding_size)
                features = features.unsqueeze(dim=1) # (N, 1, embedding size)
                embedded = self.word_embedder(features)
            else:
                last_words = torch.tensor([[self.vocab(ls[iter - 1])] for ls in captions]).to(images.device) # (N, seq_len=1)
                # embed
                embedded = self.word_embedder(last_words) # (N, 1, embedding_size)
            # (N, 1, hidden_size)
            hidden_output, (hidden_states, cell_states) = self.decoder(embedded, (hidden_states, cell_states))
            output = self.hidden2output(hidden_output) # (N, 1, vocab_size) 1-hot encoded
            output = output.squeeze(dim=1) # collapse (N, 1, vocab_size) -> (N, vocab_size)
            if deterministic: # deterministically sample from softmax
                # returns values, indices, we only want indices to decode using vocab
                _, indices = torchf.log_softmax(output, dim=1).max(dim=1) # 1d array size N
            else: # use temperature in softmax and sample
                # calc softmax w/ temperature
                softmax = torchf.softmax(output / temperature, dim=1) # (N, vocab_size)
                # sample softmax
                indices = Categorical(softmax).sample() # 1d array size N
            for i in range(batch_size):
                if keep_generating[i]:
                    word_idx = indices[i].item()
                    word = self.vocab.idx2word[word_idx]
                    captions[i][iter] = word
                    if self.vocab.idx2word[word_idx] == "<end>":
                        num_complete += 1
                        keep_generating[i] = False
            iter += 1
        # cleanup all the flag words
        return [[word for word in ls if word != "<start>" and word != "<end>" and word != "<pad>"] for ls in captions]
