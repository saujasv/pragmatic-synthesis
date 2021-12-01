import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import tqdm
import wandb
from pathlib import Path
import json

from grammar import ShapeGridGrammar
from utils import get_utterance_cache, LiteralSpeaker, spec_to_grid_one_hot
from models import TransformerListener, MLPListener

class MLPTrainer:
    def __init__(
        self,
        grammar,
        model,
        training_programs,
        batch_size,
        min_spec_len,
        max_spec_len,
        learning_rate,
        checkpoints_path,
        device='cpu'
        ):
        self.grammar = grammar
        self.n_productions = len(self.grammar.semantics)
        self.output_arity = max(len(s) for s in self.grammar.semantics)
        self.model = model
        self.training_programs = training_programs
        self.batch_size = batch_size
        self.utterance_cache = get_utterance_cache(grammar, device)
        self.min_spec_len = min_spec_len
        self.max_spec_len = max_spec_len
        self.literal_speaker = LiteralSpeaker(grammar, min_spec_len, max_spec_len)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.checkpoints_path = checkpoints_path
        self.use_wandb = use_wandb
        self.step = 0

    def training_step(self):
        prog = random.sample(self.training_programs, self.batch_size)
        n = random.randint(self.min_spec_len, self.max_spec_len)
        spec = [self.literal_speaker.get_spec(p, n) for p in prog]
        x = torch.stack([spec_to_grid_one_hot(s, self.grammar) for s in spec])
        y = self.model(x).view(self.batch_size, self.n_productions, self.output_arity)
        target = torch.tensor(prog)
        loss = sum([F.cross_entropy(y[:, i, :], target[:, i], reduction='mean') for i in range(self.n_productions)])
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def train(self, n_steps, save_every=5000, log_every=100):
        self.model.train()
        losses = list()
        for i in range(n_steps):
            loss = self.training_step()
            self.step += 1
            losses.append(loss)
            if i % log_every == 0:
                if self.use_wandb:
                    wandb.log({
                        "train/loss": sum(losses) / len(losses),
                        "trainer/step": self.step
                        })
                else:
                    print(f"TRAIN {i} loss = {sum(losses) / len(losses)}")
                losses = list()
            
            if self.step % save_every == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self):
        path = Path(self.checkpoints_path) / f'checkpoint-{self.step}.pt'
        torch.save({
            # 'config': self.model.get_config(),
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, str(path))


class TransformerTrainer:
    def __init__(
        self, 
        grammar, 
        model, 
        training_programs, 
        batch_size,
        min_spec_len,
        max_spec_len,
        learning_rate,
        validation_data,
        checkpoints_path,
        use_wandb,
        device='cpu'
    ):
        self.grammar = grammar
        self.model = model
        self.training_programs = training_programs
        self.batch_size = batch_size
        self.utterance_cache = get_utterance_cache(grammar, device)
        self.min_spec_len = min_spec_len
        self.max_spec_len = max_spec_len
        self.literal_speaker = LiteralSpeaker(grammar, min_spec_len, max_spec_len)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.checkpoints_path = checkpoints_path
        self.use_wandb = use_wandb
        self.step = 0

    def training_step(self):
        prog = random.sample(self.training_programs, self.batch_size)
        n = random.randint(self.min_spec_len, self.max_spec_len)
        spec = [self.literal_speaker.get_spec(p, n) for p in prog]
        x = torch.stack([torch.stack([self.utterance_cache[u] for u in s]) for s in spec])
        y = self.model(x)
        target = torch.tensor(prog)
        loss = sum([F.cross_entropy(prod, target[:, i], reduction='mean') for i, prod in enumerate(y)])
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def train(self, n_steps, save_every=5000, log_every=100):
        self.model.train()
        losses = list()
        for i in range(n_steps):
            loss = self.training_step()
            self.step += 1
            losses.append(loss)
            if i % log_every == 0:
                if self.use_wandb:
                    wandb.log({
                        "train/loss": sum(losses) / len(losses),
                        "trainer/step": self.step
                        })
                else:
                    print(f"TRAIN {i} loss = {sum(losses) / len(losses)}")
                losses = list()
            
            if self.step % save_every == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self):
        path = Path(self.checkpoints_path) / f'checkpoint-{self.step}.pt'
        torch.save({
            'config': self.model.get_config(),
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, str(path))

if __name__ == "__main__":
    g = ShapeGridGrammar(7)
    model = MLPListener(g, 256)

    with open('./data/train_programs.json') as f:
        training_programs = json.load(f)

    use_wandb = False

    if use_wandb:
        wandb.init(project="pragmatic-synthesis", name="literal-listener")

    trainer = MLPTrainer(g, model, training_programs, 8, 2, 25, 5e-4, './checkpoints')

    trainer.train(150000)
    trainer.save_checkpoint()