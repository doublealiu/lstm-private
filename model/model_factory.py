from base_model import BaselineEncoderDecoder
# Build and return the model here based on the configuration.
from lstm_model import LSTMModel
from new_model import Encoder_Decoder_new
from rnn_model import RNNModel


def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type'].lower()
    num_layers = config_data['model']['num_layers']

    # You may add more parameters if you want
    # model = BaselineEncoderDecoder(hidden_size, embedding_size, vocab, num_layers, model_type)
    if model_type == "lstm":
        model = BaselineEncoderDecoder(hidden_size, embedding_size, vocab, num_layers, model_type)
    elif model_type == "lstm2":
        model = LSTMModel(embedding_size, hidden_size, vocab, num_layers)
    elif model_type == "rnn":
        model = RNNModel(embedding_size, hidden_size, vocab, num_layers)
    elif model_type == "arch2":
        model = Encoder_Decoder_new(hidden_size, embedding_size, num_layers, model_type, vocab)
    else:
        model = LSTMModel(embedding_size, hidden_size, vocab, num_layers)
    return model
