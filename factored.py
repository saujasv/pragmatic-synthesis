import heapq
import json
import tqdm
import argparse
from grammar import ShapeGridGrammar
import numpy as np

def search(P, spec, grammar, search_budget=50):
    """
        Search for a program given a distribution P over each of it's 
        productions.

        :param P: Distribution over productions. 2D list of size N x A, where
                    N is the number of productions and A is the maximum arity of
                    a production
        :param spec: Spec which the program has to satisfy
        :param grammar: ShapeGridGrammar (or similar class) in which programs
                        are defined.
    """

    # Convert the 2D list to a list of dictionaries where the distribution for 
    # each production is a dictionary mapping production index to probability
    P = [{i: p for i, p in enumerate(row)} for row in P]

    # Sort the indices for each production by their probability, and discard 0
    # probability productions
    P = [
        sorted(
            [(k, np.log(v)) for k, v in prob.items() if v > 0], 
            key=lambda x: x[1], reverse=True
            ) for prob in P
            ]

    # Priority queue for programs
    searched = set()
    pq = [(-sum([prod[0][1] for prod in P]), [0 for _ in P])]
    while len(pq) > 0:
        # idx is a list of indices which defines the program as [P[j][idx[j]]] 
        # where j is the index of the production and idx[j] chooses which value
        # the production should take
        score, idx = heapq.heappop(pq)

        if len(searched) > search_budget:
            break

        hashed = '|'.join(map(str, idx))
        if hashed in searched:
            continue
        else:
            searched.add(hashed)

        # If a program consistent with the spec is found, break
        if grammar.consistent([P[i][j][0] for i, j in enumerate(idx)], spec):
            break

        for ii in range(len(P)):
            # Iterate over productions. If it is possible to move the next best 
            # value for a production, change that in idx to create new_idx and 
            # add new_idx to the queue
            new_idx = list(idx)
            if new_idx[ii] + 1 < len(P[ii]):
                new_idx[ii] += 1
                heapq.heappush(
                    pq, 
                    (-sum([P[i][j][1] for i, j in enumerate(new_idx)]), new_idx)
                    )
    
    # When the loop breaks, idx is a program consistent with spec, or best 
    # program found during search
    return [P[i][j][0] for i, j in enumerate(idx)], score

class LiteralListenerFactor:
    """
        One factor of a factored literal listener.
    """
    def __init__(self, production, grammar, version_space, uniform=True):
        # Which factor (production) of the program this listener reasons about
        self.pi = production
        self.vs = version_space
        self.grammar = grammar
        self.N = max([len(s) for s in grammar.semantics])

        # Whether to use a uniform prior or a count-based prior (uniform if 
        # True)
        self.uniform = uniform
    
    def get_distribution(self, spec, vs_cache=None):
        """
            Get distribution of values for the production self.pi.
        """
        # Order doesn't matter within this spec for listener, sort to 
        # facilitate caching
        spec = tuple(sorted(spec))

        # Check whether this spec is in cache
        if not vs_cache is None and spec in vs_cache:
            vs_us = vs_cache[spec]
        else:
            atoms = [','.join(map(str, u)) for u in spec]
            vs_us = set(self.vs['spec'][atoms[0]])
            for u in atoms[1:]:
                vs_us = vs_us.intersection(self.vs['spec'][u])
            if not vs_cache is None:
                vs_cache[spec] = vs_us

        P = [0 for p in range(self.N)]
        # Iterate over programs that satisfy spec (fetched from version space)
        for pid in vs_us:
            for i, p in enumerate(self.vs['production_counts'][pid][self.pi]):
                # If uniform, counts are 1-0, and if not, counts as by number
                # of programs that satisfy the spec and have i as the value for
                # production self.pi
                P[i] += (1 if p > 0 else 0) if self.uniform else p
        
        # Normalizing factor
        norm = sum(P)

        # If all 0, return as is, else return normalized distribution
        if norm == 0:
            return P
        else:
            return [p / norm for p in P]

class FactoredLiteralListener:
    """
        Factored literal listener that reasons over individual factors and 
        combines results.
    """
    def __init__(self, grammar, version_space, uniform=True):
        self.grammar = grammar
        self.vs = version_space
        # Define each factor of the listener
        self.factors = [
            LiteralListenerFactor(
                i, grammar, version_space, uniform
                ) for i in range(len(grammar.semantics))
                ]
    
    def get_distribution(self, spec):
        """
            Get list of distributions over each factor of the program

            :param spec: list of spec elements (i, j, object)
        """
        # Output distribution is the distribution obtained from each factor
        return [f.get_distribution(spec) for f in self.factors]
    
    def synthesize_programs(self, spec):
        """
            Find best program that satisfies the spec.

            :param spec: list of spec elements (i, j, object)
        """

        P = self.get_distribution(spec)
        return search(P, spec, self.grammar)

class PragmaticSpeakerFactor:
    """
        One factor of a factored pragmatic speaker.
    """
    def __init__(self, production, grammar, version_space, uniform=True):
        # Which factor (production) of the program this speaker reasons about
        self.pi = production

        # Literal listener to recursively reason over the same factor
        self.literal_listener = LiteralListenerFactor(
            production, grammar, version_space, uniform
            )
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
        for u in spec_space:
            spec = tuple(history + [u])
            if not L0_cache is None and spec in L0_cache:
                P_L0 = L0_cache[spec]
            else:
                P_L0 = self.literal_listener.get_distribution(spec)
                if not L0_cache is None:
                    L0_cache[spec] = P_L0
            P_S1[u] = P_L0[p]
        norm = sum(P_S1.values())
        if norm == 0:
            return P_S1
        else:
            return {u: p / norm for u, p in P_S1.items()}

class PragmaticListenerFactor:
    """
        One factor of a factored pragmatic speaker.
    """
    def __init__(self, production, grammar, version_space, uniform=True):
        # Which factor (production) of the program this listener reasons about
        self.pi = production
        self.N = max([len(s) for s in grammar.semantics])

        # Pragmatic speaker to recursively reason over the same factor
        self.pragmatic_speaker = PragmaticSpeakerFactor(
            production, grammar, version_space, uniform
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

class FactoredPragmaticListener:
    """
        Factored pragmatic listener that recursively reasons over individual 
        factors and combines results.
    """
    def __init__(self, grammar, version_space, uniform=True):
        self.grammar = grammar
        # Define each factor of the listener
        self.factors = [
            PragmaticListenerFactor(
                i, grammar, version_space, uniform
                ) for i in range(len(grammar.semantics))
            ]
    
    def get_distribution(self, spec):
        """
            Get list of distributions over each factor of the program

            :param spec: list of spec elements (i, j, object)
        """
        return [f.get_distribution(spec) for f in self.factors]
    
    def synthesize_programs(self, spec):
        """
            Find best program that satisfies the spec.

            :param spec: list of spec elements (i, j, object)
        """
        P = self.get_distribution(spec)
        return search(P, spec, self.grammar)

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
        "--uniform",
        "-u",
        action="store_true",
        help="Use a uniform prior for literal listener"
        )
    parser.add_argument(
        "--version_space", 
        type=str, 
        default='version_spaces/factored_version_space.json',
        help="Path to version space file"
        )

    args = parser.parse_args()
    with open(args.version_space, 'r') as f:
        vs = json.load(f)

    g = ShapeGridGrammar(7)
    if not args.listener is None:
        with open(args.specs, 'r') as f:
            data = json.load(f)
        
        specs = [[tuple(u) for u in s] for p, s in data]
        programs = [p for p, s in data]
        
        if args.listener == "literal":
            listener = FactoredLiteralListener(
                g, vs, uniform=args.uniform
                )
        else:
            listener = FactoredPragmaticListener(
                g, vs, uniform=args.uniform
                )

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

            # Write results to file at intermediate points
            if i % args.write_every:
                with open(args.outputs, 'w') as f:
                    json.dump(outputs, f)

        with open(args.outputs, 'w') as f:
            json.dump(outputs, f)
