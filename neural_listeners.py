import torch
import torch.nn.functional as F
import json
import argparse
import tqdm
from models import MLPListener
from utils import spec_to_grid_one_hot
from factored import search
from grammar import ShapeGridGrammar

class NeuralPragmaticSpeakerFactor:
    """
        One factor of a factored pragmatic speaker.
    """
    def __init__(self, production, grammar, model):
        # Which factor (production) of the program this speaker reasons about
        self.pi = production

        # Literal listener to recursively reason over the same factor
        self.literal_listener = model
        self.grammar = grammar
    
    def step(self, p, history, L0_cache=None):
        """
            Generate the next utterance given history when speaking about a 
            program where the production self.pi takes value p

            :param p: integer value of production self.pi
            :param history: list of spec items
        """

        # Find squares that are covered in the history (no need to enumerate
        # all possibilities for those)
        covered = [(i, j) for i, j, r in history]
        spec_space = set()
        for i in range(self.grammar.grid_size):
            for j in range(self.grammar.grid_size):
                if (i, j) in covered:
                    continue
                
                # For cells that are not covered, we add all possible utterances
                # to spec space that is reasoned about
                spec_space.add((i, j, self.grammar.shapes[-1]))
                for s in self.grammar.shapes[:-1]:
                    for c in self.grammar.colours:
                        spec_space.add((i, j, s + c))
        
        # For each element of the spec space, get the literal listener 
        # distribution
        P_S1 = dict()
        specs = list()
        new_utterances = list()
        for u in spec_space:
            spec = tuple(history + [u])
            if not L0_cache is None and spec in L0_cache:
                P_L0 = L0_cache[spec][self.pi]
                P_S1[u] = P_L0[p]
            else:
                new_utterances.append(u)
                specs.append(spec)

        if len(new_utterances) > 0:
            x = torch.stack([spec_to_grid_one_hot(s, self.grammar) for s in specs])
            y = F.softmax(self.literal_listener(x).view(len(specs), 12, 7), dim=2) #.tolist()
            y = y.tolist()
            for i, u in enumerate(new_utterances):
                L0_cache[specs[i]] = {j: y[i][j] for j in range(12)}
                P_S1[u] = L0_cache[specs[i]][self.pi][p]
        
        norm = sum(P_S1.values())

        if norm == 0:
            return P_S1
        else:
            return {u: p / norm for u, p in P_S1.items()}

class NeuralPragmaticListenerFactor:
    """
        One factor of a factored pragmatic speaker.
    """
    def __init__(self, production, grammar, model):
        # Which factor (production) of the program this listener reasons about
        self.pi = production
        self.N = len(grammar.semantics[production])

        # Pragmatic speaker to recursively reason over the same factor
        self.pragmatic_speaker = NeuralPragmaticSpeakerFactor(
            production, grammar, model
            )

        self.L0_cache = dict()
        self.CACHE_SIZE = 5000

    def get_distribution(self, spec):
        # Break spec into history + last element, and have the pragmatic 
        # speaker generate distributions over utterances based on history from 
        # which the probability corresponding to the last element is chosen as 
        # the probability for the production self.pi taking the value p
        P = [self.pragmatic_speaker.step(p, spec[:-1], self.L0_cache)[spec[-1]] for p in range(self.N)]

        if len(self.L0_cache) > self.CACHE_SIZE:
            self.L0_cache = dict()

        norm = sum(P)

        # If all 0, return as is, else return normalized distribution
        if norm == 0:
            return P
        else:
            return [p / norm for p in P]

class NeuralFactoredPragmaticListener:
    """
        Factored pragmatic listener that recursively reasons over individual 
        factors and combines results.
    """
    def __init__(self, grammar, model):
        self.grammar = grammar
        # Define each factor of the listener
        self.factors = [
            NeuralPragmaticListenerFactor(i, grammar, model) 
            for i in range(len(grammar.semantics))
            ]
    
    def get_distribution(self, spec):
        """
            Get list of distributions over each factor of the program

            :param spec: list of spec elements (i, j, object)
        """
        return [f.get_distribution(spec) for f in self.factors]

class NeuralFactoredLiteralListener:
    """
        Factored pragmatic listener that recursively reasons over individual 
        factors and combines results.
    """
    def __init__(self, grammar, model):
        self.grammar = grammar
        # Define each factor of the listener
        self.model = model
    
    def get_distribution(self, spec):
        """
            Get list of distributions over each factor of the program

            :param spec: list of spec elements (i, j, object)
        """
        return F.softmax(self.model(spec_to_grid_one_hot(spec, self.grammar)).view(1, 12, 7), dim=2)[0].tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--listener", 
        type=str, 
        choices=["literal", "pragmatic"],
        help="Type of listener"
        )
    parser.add_argument(
        "--specs", 
        type=str,
        help="Path to JSON file containing pairs of (program, spec) as a list"
        )
    parser.add_argument(
        "--outputs", 
        type=str,
        help="Path to JSON file to write outputs"
        )
    parser.add_argument(
        "--write_every", 
        type=int,
        default=20,
        help="Write to JSON file every N times"
        )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="Path to checkpoint"
        )

    args = parser.parse_args()

    g = ShapeGridGrammar(7)
    model = MLPListener(g, 256)

    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])

    if not args.listener is None:
        with open(args.specs, 'r') as f:
            data = json.load(f)
        
        specs = [[tuple(u) for u in s] for p, s in data]
        programs = [p for p, s in data]
        
        if args.listener == "literal":
            outputs = list()
            for i, spec in tqdm.tqdm(enumerate(specs)):
                expt = {
                    'ground_truth_program': programs[i],
                    'spec': spec,
                    'synthesis': list()
                }
                for j in range(1, len(spec) + 1):
                    try:
                        p = F.softmax(model(spec_to_grid_one_hot(spec[:j], g)).view(1, 12, 7), dim=2)[0].tolist()
                        best_prog = search(p, spec[:j], g)
                        expt['synthesis'].append({
                            "distribution": p,
                            "top_program": best_prog
                        })
                    except IndexError:
                        expt['synthesis'].append({
                            "distribution": p
                        })
                outputs.append(expt)
        else:
            listener = NeuralFactoredPragmaticListener(g, model)

            outputs = list()
            for i, spec in tqdm.tqdm(enumerate(specs)):
                expt = {
                    'ground_truth_program': programs[i],
                    'spec': spec,
                    'synthesis': list()
                }
                for j in range(1, len(spec) + 1):
                    try:
                        p = listener.get_distribution(spec[:j])
                        best_prog = search(p, spec[:j], g)
                        expt['synthesis'].append({
                            "distribution": p,
                            "top_program": best_prog
                        })
                    except IndexError:
                        expt['synthesis'].append({
                            "distribution": p
                        })
                outputs.append(expt)

        with open(args.outputs, 'w') as f:
            json.dump(outputs, f, indent=2)