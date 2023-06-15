import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as TF
import torch

from vocab import Vocabulary


class Resnet50Encoder(nn.Module):
    def __init__(self, embedding_size, freeze_encoder=True):
        self.freeze_encoder = freeze_encoder
        super(Resnet50Encoder, self).__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        if self.freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False
        resnet.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        # self.softmax = nn.Softmax(dim=1) do we need this?

        self.resnet = resnet
        # self.relu = nn.ReLU(inplace=True) do we need this?

    def forward(self, images):
        out = self.resnet(images)
        return out
    

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, num_layers, model_type):
        super(Decoder, self).__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_size)
        self.fc         = nn.Linear(hidden_size, vocab_size)
        if model_type == 'RNN':
            self.recurrent = nn.RNN(input_size = embedding_size, hidden_size = hidden_size, num_layers = num_layers)
        else:
            self.recurrent = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, num_layers = num_layers)

    def forward(self, features, captions):
        features   = features.unsqueeze(1)
        embeddings = self.embedding(captions)
        embeddings = torch.cat(features, embeddings[:,:-1,:], dim=1)
        hiddens, c = self.recurrent(embeddings)
        return self.fc(hiddens)


class BaselineEncoderDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab: Vocabulary, num_layers, model_type):
        vocab_size = len(vocab)
        super(BaselineEncoderDecoder, self).__init__()
        self.model_type = "LSTM"
        self.encoder = Resnet50Encoder(embedding_size)
        self.decoder = Decoder(hidden_size, embedding_size, vocab_size, num_layers, model_type)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs