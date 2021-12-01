# Efficient Pragmatic Program Synthesis with Informative Specifications

Code for enumerative and neural synthesizers, with pragmatic listeners build on top.

To train a neural literal listener:
```
python train.py -h
```
Training programs used are in `data/train_programs.json`.

To run experiments and get results:
```
python experiment.py -h
```
Here, you can specify whether the listener type (neural/enumerative), listener variant (literal/pragmatic), and provide a path to one of the JSON files in `specs` which contain specifications. Results and logs are written to the `output` file.