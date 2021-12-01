import argparse
import json
from grammar import ShapeGridGrammar
from factored import search, FactoredLiteralListener, FactoredPragmaticListener
from neural_listeners import NeuralFactoredLiteralListener, NeuralFactoredPragmaticListener
import torch
from models import MLPListener
import tqdm

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--variant", 
    type=str, 
    choices=["literal", "pragmatic"],
    help="Type of listener"
    )
parser.add_argument(
    "--type", 
    type=str,
    help="neural/enumerative",
    choices=['neural', 'enumerative']
    )
parser.add_argument(
    "--checkpoint",
    "-c",
    type=str,
    help="Path to checkpoint (if neural)"
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
g = ShapeGridGrammar(7)
with open(args.version_space, 'r') as f:
    vs = json.load(f)

if args.type == "neural":
    model = MLPListener(g, 256)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    if args.variant == "literal":
        listener = NeuralFactoredLiteralListener(g, model)
    else:
        listener = NeuralFactoredPragmaticListener(g, model)
else:
    if args.variant == "literal":
        listener = FactoredLiteralListener(g, vs, uniform=False)
    else:
        listener = FactoredPragmaticListener(g, vs, uniform=False)

with open(args.specs, 'r') as f:
    data = json.load(f)
    
specs = [[tuple(u) for u in s] for p, s in data]
programs = [p for p, s in data]

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
            best_prog, score = search(p, spec[:j], g)
            expt['synthesis'].append({
                "distribution": p,
                "top_program": best_prog,
                "score": score,
                "correct": g.hash(best_prog) == g.hash(programs[i])
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