from collections import defaultdict
import itertools
import json

class Grammar:
    def __init__(self, variables, productions, semantics):
        # List of variables in the grammar
        self.idx2variable = variables

        # map from variables to index in idx2variable
        self.variable2idx = {v: i for i, v in enumerate(variables)}

        # List of tuples, with each tuple containing only the 
        # variables on the right hand side of the production
        self.productions = productions

        # List of tuples of functions, where the jth element of a tuple at index
        # i represents the semantics of the jth production with the ith variable as
        # the LHS
        self.semantics = semantics

    def execute(self, program_vector, x, y):
        # Start execution at the start node of the program
        return self.__execute_helper(program_vector, x, y, 0)
    
    def __execute_helper(self, program_vector, x, y, idx):
        # Execute the program by recursively computing the values of the arguments to the ith production,
        # and applying the semantics of the ith production to obtain the result.
        args = [self.__execute_helper(program_vector, x, y, self.variable2idx[v]) 
                    for v in self.productions[idx]]
        return self.semantics[idx][program_vector[idx]](x, y, *args)
    
    def valid(self, program_vector):
        raise NotImplementedError
    
    def enumerate(self):
        # return self.__enumerate_helper(0, [])
        for prog in self.__enumerate_helper(0):
            if not self.valid(prog):
                continue
            else:
                yield prog
    
    def __enumerate_helper(self, i):
        if i == len(self.productions) - 1:
            for j in range(len(self.semantics[i])):
                yield [j]
        else:
            for partial_prog in self.__enumerate_helper(i + 1):
                for j in range(len(self.semantics[i])):
                    yield [j, *partial_prog]

    
    def sat(self, program, spec):
        return all([self.execute(program, x, y) == (s, c) for x, y, s, c in spec])

def make_constant_function(i):
    return lambda x, y: i

class ShapeGridGrammar(Grammar):
    """
        Program vector represents the program as a list of indices, where the ith
        element of the vector indicates the index of the production of the ith variable below.

        Program -> Shape, Colour
        Shape -> Box(Left, Right, Top, Bottom, Thickness, Outside, Inside)
        Left -> 0 | 1 | 2 | 3 | ... | grid_size
        Right -> 0 | 1 | 2 | 3 | ... | grid_size
        Top -> 0 | 1 | 2 | 3 | ... | grid_size
        Bottom -> 0 | 1 | 2 | 3 | ... | grid_size
        Thickness -> 1 | 2 | 3
        O   -> chicken | pig (or all the shapes in the shapes argument except the last)
        I   -> chicken | pig | pebble (or all the shapes in the shapes argument)
        Colour   -> [red , green , blue][A2(A1)]
        A1 -> x | y | x + y
        A2 -> lambda z:0 | lambda z:1 | lambda z:2 | lambda z:z%2 |lambda z:z%2+1 | lambda z:2*(z%2)
    """

    def __init__(
        self, 
        grid_size, 
        shapes=['C', 'S', 'E'], 
        colours=['R', 'G', 'B'],
        empty = 'E'
        ):
        variables = ['Program', 'Shape', 'Left', 'Right', 'Top', 'Bottom', 'Thickness', 'Outside',
                        'Inside', 'Colour', 'A1', 'A2']
        productions = [('Shape', 'Colour'), 
                        ('Left', 'Right', 'Top', 'Bottom', 'Thickness', 'Outside', 'Inside'), 
                        tuple(), tuple(), tuple(), tuple(), tuple(), tuple(), tuple(),
                        ('A1', 'A2'),
                        tuple(), tuple()]
        semantics = [
            (lambda x, y, shape, colour: (shape, colour),),
            (lambda x, y, left, right, top, bottom, thickness, outside, inside: 
                empty if (left > x or x > right or top > y or y > bottom)
                        else (outside if any([
                            0 <= (x - left) < thickness, 
                            0 <= (y - top) < thickness, 
                            0 <= (right - x) < thickness, 
                            0 <= (bottom - y) < thickness]
                            )
                            else inside),),
            tuple(make_constant_function(i) for i in range(grid_size)), 
            tuple(make_constant_function(i) for i in range(grid_size)), 
            tuple(make_constant_function(i) for i in range(grid_size)), 
            tuple(make_constant_function(i) for i in range(grid_size)),
            tuple(make_constant_function(i + 1) for i in range(3)),
            tuple(make_constant_function(shape) for shape in shapes[:-1]),
            tuple(make_constant_function(shape) for shape in shapes),
            (lambda x, y, a1, a2: colours[a2(a1)],),
            (lambda x, y: x, lambda x, y: y, lambda x, y: x + y),
            (lambda x, y: lambda z : 0,
                lambda x, y: lambda z : 1,
                lambda x, y: lambda z : 2,
                lambda x, y: lambda z : 0 if z % 2 else 1,
                lambda x, y: lambda z : 1 if z % 2 else 2,
                lambda x, y: lambda z : 2 if z % 2 else 0,
                )
        ]

        super().__init__(variables, productions, semantics)
        self.grid_size = grid_size
        self.shapes = shapes
        self.colours = colours
        self.shape2idx = {shape: i for i, shape in enumerate(self.shapes)}
        self.colour2idx = {colour: i for i, colour in enumerate(self.colours)}
    
    def valid(self, program_vector):
        program, shape, left, right, top, bottom, thickness, outside, inside, colour, a1, a2 = program_vector
        return all(
            [program_vector[i] < len(s) for i, s in enumerate(self.semantics)]
            ) and right - left > 1 and bottom - top > 1 \
                and left + 2 * thickness - 1 <= right \
                and top + 2 * thickness - 1 <= bottom
    
    def consistent(self, prog, spec):
        if not self.valid(prog):
            return False
        result = [(i, j, self.make_atom(self.execute(prog, i, j))) for j in range(self.grid_size) for i in range(self.grid_size)]
        return len(set(spec).difference(result)) == 0
    
    def make_atom(self, result):
        if result[0] == self.shapes[-1]:
            return result[0]
        else:
            return result[0] + result[1]
    
    def print(self, prog):
        print(*['|'.join([f"{self.make_atom(self.execute(prog, i, j)):2s}" for j in range(self.grid_size)]) for i in range(self.grid_size)], sep='\n')


    def hash(self, prog):
        result = [
            [(i, j, self.execute(prog, i, j)) for j in range(self.grid_size)] 
            for i in range(self.grid_size)
            ]
        return '|'.join(['.'.join([s if s == self.shapes[-1] else s + c for i, j, (s, c) in row]) for row in result])
    
    def unhash(self, hashed):
        shapes = set()
        for i, row in enumerate(hashed.split('|')):
            for j, cell in enumerate(row.split('.')):
                shapes.add((i, j, cell))
        return shapes

if __name__ == "__main__":
    grammar = ShapeGridGrammar(7)
    program_space = defaultdict(list)
    i = 0
    for prog in grammar.enumerate():
        result = [[(i, j, grammar.make_atom(grammar.execute(prog, i, j))) for j in range(7)] for i in range(7)]
        resultstr = '|'.join(['.'.join([r for i, j, r in row]) for row in result])
        program_space[resultstr].append(prog)
        i += 1
    
    vs = defaultdict(list)
    production_counts = list()
    for progid, (k, progs) in enumerate(program_space.items()):
        counts = [[0 for i in range(7)] for j in range(12)]
        for prog in progs:
            for pi, prod in enumerate(prog):
                counts[pi][prod] += 1
        production_counts.append(counts)
        result = [(i, j, grammar.execute(progs[0], i, j)) for i in range(7) for j in range(7)]
        for i, j, r in result:
            atom = "{},{},{}".format(i, j, grammar.make_atom(r))
            vs[atom].append(progid)
    
    with open("version_spaces/factored_version_space.json", 'w') as f:
        json.dump({"programs": list(program_space.keys()), "targets": list(program_space.values()), "spec": vs, "production_counts": production_counts}, f)
    
    print(len(program_space))