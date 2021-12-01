import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import random
import tqdm
import wandb

from grammar import ShapeGridGrammar
from utils import get_utterance_cache, LiteralSpeaker

class TransformerListener(nn.Module):
    def __init__(
        self, 
        grammar, 
        hidden_size=128, 
        n_encoder_heads=2,
        n_encoder_layers=2,
        encoder_ff_size=512,
        encoder_dropout=0.1,
        n_decoder_heads=1,
        n_decoder_layers=1,
        decoder_ff_size=512,
        decoder_dropout=0.1
    ):
        super(TransformerListener, self).__init__()
        self.input_size = 2 * grammar.grid_size + len(grammar.shapes) + len(grammar.colours)
        # self.input_size = 2 + len(grammar.shapes) + len(grammar.colours)
        self.hidden_size = hidden_size
        self.n_encoder_heads = n_encoder_heads
        self.n_encoder_layers = n_encoder_layers
        self.encoder_ff_size = encoder_ff_size
        self.encoder_dropout = encoder_dropout
        self.n_decoder_heads = n_decoder_heads
        self.n_decoder_layers = n_decoder_layers
        self.decoder_ff_size = decoder_ff_size
        self.decoder_dropout = decoder_dropout

        self.embedding = nn.Linear(self.input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_encoder_heads,
            dim_feedforward=encoder_ff_size,
            dropout=encoder_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.n_productions = len(grammar.semantics)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_decoder_heads,
            dim_feedforward=decoder_ff_size,
            dropout=decoder_dropout,
            batch_first=True
        )
        self.transformer_decoders = [
            nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers) 
            for _ in range(self.n_productions)
            ]

        self.output_arity = max(len(s) for s in grammar.semantics)
        self.output_projections = [nn.Linear(hidden_size, len(s)) if len(s) > 1 else None for s in grammar.semantics]
    
    def forward(self, spec):
        embedded_spec = self.embedding(spec)
        encoded = self.transformer_encoder(embedded_spec)
        decoded = [decoder(encoded).sum(dim=1) for decoder in self.transformer_decoders]
        # output = [F.softmax(self.output_projections[i](d), dim=1) for i, d in enumerate(decoded)]
        output = [
            self.output_projections[i](d) if not self.output_projections[i] is None else torch.ones(spec.size(0), 1)
            for i, d in enumerate(decoded)
            ]
        return output
    
    @classmethod
    def from_config(config, grammar):
        return TransformerListener(
            grammar, 
            hidden_size=config['hidden_size'], 
            n_encoder_heads=config['n_encoder_heads'],
            n_encoder_layers=config['n_encoder_layers'],
            encoder_ff_size=config['encoder_ff_size'],
            encoder_dropout=config['encoder_dropout'],
            n_decoder_heads=config['n_decoder_heads'],
            n_decoder_layers=config['n_decoder_layers'],
            decoder_ff_size=config['decoder_ff_size'],
            decoder_dropout=config['decoder_dropout']
        )
    
    def get_config(self):
        return {
            'hidden_size': self.hidden_size,
            'n_encoder_heads': self.n_encoder_heads,
            'n_encoder_layers': self.n_encoder_layers,
            'encoder_ff_size': self.encoder_ff_size,
            'encoder_dropout': self.encoder_dropout,
            'n_decoder_heads': self.n_decoder_heads,
            'n_decoder_layers': self.n_decoder_layers,
            'decoder_ff_size': self.decoder_ff_size,
            'decoder_dropout': self.decoder_dropout
        }
    
class MLPListener(nn.Module):
    def __init__(
        self,
        grammar,
        hidden_size
    ):
        super(MLPListener, self).__init__()
        self.input_size = 6 * (grammar.grid_size ** 2)
        self.hidden_size = hidden_size
        self.output_size = len(grammar.semantics) * max(len(s) for s in grammar.semantics)

        self.network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
    
    def forward(self, x):
        return self.network(x)