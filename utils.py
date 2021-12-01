import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from grammar import ShapeGridGrammar

def utterance2vec(u, grammar, device='cpu'):
    D = 2 * grammar.grid_size + len(grammar.shapes) + len(grammar.colours)
    X = torch.zeros(D, device=device)
    x, y, r = u
    X[x] = 1
    X[grammar.grid_size + y] = 1
    X[2 * grammar.grid_size + grammar.shape2idx[r[0]]] = 1
    if r != 'E':
        X[2 * grammar.grid_size + len(grammar.shapes) + grammar.colour2idx[r[1]]] = 1

    # D = 2 + len(grammar.shapes) + len(grammar.colours)
    # X = torch.zeros(D, device=device)
    # x, y, r = u
    # X[0] = x
    # X[1] = y
    # X[2 + grammar.shape2idx[r[0]]] = 1
    # if r != 'E':
    #     X[2 + len(grammar.shapes) + grammar.colour2idx[r[1]]] = 1

    return X

def get_utterance_cache(grammar, device='cpu'):
    cache = dict()
    for x in range(grammar.grid_size):
        for y in range(grammar.grid_size):
            r = grammar.shapes[-1]
            cache[(x, y, grammar.shapes[-1])] = utterance2vec((x, y, r), grammar, device)
            for s in grammar.shapes[:-1]:
                for c in grammar.colours:
                    r = s + c
                    cache[(x, y, r)] = utterance2vec((x, y, r), grammar, device)

    return cache

SHAPE2IDX = {
    'C': 1,
    'S': 2,
    'E': 3
}

COLOUR2IDX = {
    'R': 1,
    'G': 2,
    'B': 3,
    'E': 4
}

def spec_to_grid(spec, grammar, device='cpu'):
    N = grammar.grid_size
    shapes = torch.zeros((N, N), device=device)
    colours = torch.zeros((N, N), device=device)
    for i, j, r in spec:
        shapes[i][j] = SHAPE2IDX[r[0]]
        if len(r) > 1:
            colours[i][j] = COLOUR2IDX[r[1]]
        else:
            colours[i][j] = COLOUR2IDX[r[0]]
    
    return torch.cat((shapes.view(-1, N * N), colours.view(-1, N * N)), dim=1)

def spec_to_grid_one_hot(spec, grammar, device='cpu'):
    x = torch.zeros((grammar.grid_size, grammar.grid_size, 6))
    for i, j, r in spec:
        x[i][j][SHAPE2IDX[r[0]] - 1] = 1
        if len(r) > 1:
            x[i][j][COLOUR2IDX[r[1]] - 1] = 1
    
    return x.view(1, -1)

class LiteralSpeaker:
    def __init__(self, grammar, min_spec_len, max_spec_len):
        self.grammar = grammar
        self.min_spec_len = min_spec_len
        self.max_spec_len = max_spec_len
    
    def get_spec(self, prog, n=None):
        length = random.randint(self.min_spec_len, self.max_spec_len) if n is None else n
        spec_space = [
            (x, y, self.grammar.make_atom(self.grammar.execute(prog, x, y)))
            for x in range(self.grammar.grid_size) 
            for y in range(self.grammar.grid_size)
            ]

        return random.sample(spec_space, length)