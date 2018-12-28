# Tetris AI v0

import random
from math import floor, ceil
from constants import *
from tkinter import *
from itertools import permutations as permutate

# +------------------------------
# Helper Functions
# +------------------------------

def _make_child(parent1, parent2, factor):
    '''Helps the make_child function inside the Tetris Class create a child
    genome using a weighted average between the fitnesses of the two parents.
    Adds a little randomness to avoid a value equaling zero
    
    Arguments:
        parent1: a parent genome dict, usually a random one from the genome
        parent2: a parent genome dict, usually an elite genome
        factor: a string representing the feature that will be computed
    
    Return Value:
        An integer value representing the numeric value of the genome
    '''
    fitness1 = parent1['fitness']
    fitness2 = parent2['fitness']
    factor1 = parent1[factor]
    factor2 = parent2[factor]
    
    normalizingFitness = fitness1 + fitness2
    newFitness = (fitness1 * factor1 + fitness2 * factor2) / normalizingFitness
    
    return newFitness

def transpose(matrix):
    '''Takes in a list of lists/matrix and outputs the transposed matrix.
    
    Arguments:
        matrix: a list of lists
    
    Return Value:
        a list of lists that represent the transposition of the inputted matrix
    '''
    transposedMatrix = list(map(list, zip(*matrix)))
    
    return transposedMatrix

def _rotate(shape, rotations):
    '''Helps the rotate_shape function rotate a tetris piece 90 degrees
    clockwise.
    
    Arguments:
        shape: a nxn square list of lists that represents the tetris shape we
        want to rotate
        rotations: an integer that represents the number of times we
        want to rotate the shape
    
    Return Value:
        a tetris shape that has been rotated the desired number of times
    '''
    output = shape.copy()
    for _ in range(rotations % 4):
        output = transpose(output)
        for row in output:
            row = row.reverse()
    return output

# +------------------------------
# Tetris Class
# +------------------------------
class Tetris:
    '''Instances of this class will allow the user to play Tetris with the
    computer interactively'''
    
    def __init__(self, master):
        '''Initialize the game.'''
        # Tetris board
        self.grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.master = master
        self.master.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')
        self.master.bind('<Key>', self.key_handler)
        self.canvas = Canvas(self.master, width=WINDOW_WIDTH, \
                             height=WINDOW_HEIGHT)
        self.canvas.pack(expand=YES, fill=BOTH)
        t = self.canvas.create_text(MOVEX, 36, font=FONT, text='Move: ')
        t1 = self.canvas.create_text(INDIVIDUALX, 36, font=FONT, \
                                     text='Individual: ')
        t2 = self.canvas.create_text(GENERATIONX, 36, font=FONT, \
                                     text='Generation: ')
        t3 = self.canvas.create_text(SCOREX, 36, font=FONT, text='Score: ')
        self.values = []
        self.textID = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        
        # Block setup
        self.shapes = {'I': [[0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],
                  'J': [[2, 0, 0],
                        [2, 2, 2],
                        [0, 0, 0]],
                  'L': [[0, 0, 3],
                        [3, 3, 3],
                        [0, 0, 0]],
                  'O': [[4, 4],
                        [4, 4]],
                  'S': [[0, 5, 5],
                        [5, 5, 0],
                        [0, 0, 0]],
                  'T': [[0, 6, 0],
                        [6, 6, 6],
                        [0, 0, 0]],
                  'Z': [[7, 7, 0],
                        [0, 7, 7],
                        [0, 0, 0]]}
        self.colors = ['#F92338', '#C973FF', '#1C76BC', \
                       '#FEE356', '#53D504', '#36E0FF', '#F8931D']
        
        self.currentShape = {'xCoord': 0, 'yCoord': 0, 'shape': None}
        self.onDeck = None          # List of lists/matrix of next shape
        self.storage = []
        self.storageIndex = 0
        
        # Game Values
        self.score = 0
        self.scoreID = [0, 0, 0, 0]
        self.speed = 1
        self.changeSpeed = False
        self.speeds = [1, 10, 50, 100, 500]
        self.speedIndex = 0
        self.ai = True
        self.testing = False
        self.terminal = True
        self.simp = True
        self.draw = True
        self.movesTaken = 0
        
        # algorithm for the moves
        self.moveAlgorithm = {}
        
        # evolutionary values
        self.populationSize = POPULATION_SIZE
        self.genomes = []
        self.currentGenome = -1
        self.generation = 1
        self.archive = {'populationSize': 0,
                        'currentGeneration': 0,
                        'elites': [],
                        'genomes': []}
        self.averageFitness = []
        
        # rate of mutation per occurrence
        MUTATION_RATE = 0.05
        # rate of mutations that happen
        MUTATION_STEP = 0.2
        
        self.done = False
    
    def key_handler(self, event):
        '''Handles key presses.
        
        Arguments:
            event: an event that represents the key pressed
        
        Return Value:
            none
        '''
        key = event.keysym
        
        if key == 'Down':
            self.move_down()
        elif key == 'Up':
            self.rotate_shape()
        elif key == 'Left':
            self.move_left()
        elif key == 'Right':
            self.move_right()
        elif key.upper() == 'A':
            self.ai = not self.ai
        elif key.upper() == 'U':
            if self.speedIndex < len(self.speeds) - 1:
                self.speedIndex += 1
                self.speed = self.speeds[self.speedIndex]
        elif key.upper() == 'I':
            if self.speedIndex > 0:
                self.speedIndex -= 1
                self.speed = self.speeds[self.speedIndex]
        elif key.upper() == 'S':
            self.saveState = self.get_state()
        elif key.upper() == 'L':
            self.load_state(self.saveState)
        elif key.upper() == 'Q':
            tetris.done = True
            quit()
        
        self.output()
        
    def start_loop(self):
        '''The main loop'''
        
        if self.speed == 0:
            self.draw = False
            for _ in range(3):
                self.update()
        else:
            self.draw = True
        
        self.update()
        
        if self.speed == 0:
            self.draw = True
            self.update_score()
    
    def load_state(self, state):
        '''Loads a current state. Used for jumping back to previous states
        in the breadth-first search.
        
        Arguments:
            state: a dictionary representing the state (grid, currentShape,
            onDeck, storage, storageIndex, score) that will be loaded
        
        Return Value:
            none
        '''
        # Set internal values to those of the inputted state
        self.grid = state['grid']
        self.currentShape = state['currentShape']
        self.onDeck = state['onDeck']
        self.storage = state['storage']
        self.storageIndex = state['storageIndex']
        self.score = state['score']
        
        self.output()
        self.update_score()        
    
    def update(self):
        '''Update the fitness function, makes a move (what shape to play,
        how to rotate it), and evaluate the next move.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        if self.ai == True and self.currentGenome != -1:
            results = self.move_down()
            
            # No movement means a loss or hit the bottom
            if not results['moved']:
                if results['lose']:
                    self.genomes[self.currentGenome]['fitness'] = self.score
                    self.evaluate_next_genome()
                else:
                    self.make_next_move()
        else:
            self.move_down()
        
        # Display updated score
        if len(self.grid) == 20:
            self.output()
            self.update_score()
    
    def update_score(self):
        '''Updates the score of the game.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        self.delete_score()
        if self.terminal == False:
            t0 = self.canvas.create_text(MOVEX + 50, 36, font=FONT, \
                                         text=self.movesTaken)
            t1 = self.canvas.create_text(INDIVIDUALX + 50, 36, font=FONT, \
                                         text=self.currentGenome + 1)
            t2 = self.canvas.create_text(GENERATIONX + 50, 36, font=FONT, \
                                         text=self.generation)
            t3 = self.canvas.create_text(SCOREX + 50, 36, font=FONT, \
                                        text=self.score)
            self.scoreID[0] = t0
            self.scoreID[1] = t1
            self.scoreID[2] = t2
            self.scoreID[3] = t3
        elif not self.simp:
            print(f'[Moves Taken: {self.movesTaken}, \
                     Current Genome: {self.currentGenome + 1}, \
                     Current Generation: {self.generation}, \
                     Score: {self.score}]')
    
    def make_next_move(self):
        '''Make the next move by picking a new piece. Considers if move
        exceeds our established max limit. Otherwise, make next move.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        self.movesTaken += 1
        if self.movesTaken > MOVE_LIMIT:
            self.genomes[self.currentGenome]['fitness'] = self.score
            self.evaluate_next_genome()
        else:
            oldDraw = self.draw
            self.draw = False
            possibleMoves = self.possible_moves()
            lastState = self.get_state()
            self.next_shape()
            
            # This doesn't make sense. Is it even necessary?
            for move in possibleMoves:
                nextMove = self.get_best_move(possibleMoves)
                move['rating'] += nextMove['rating']
            
            self.load_state(lastState)
            move = self.get_best_move(possibleMoves)
            
            # Manipulate the shape
            for _ in range(move['rotation']):
                self.rotate_shape()
                
            if move['translation'] < 0:
                for _ in range(abs(move['translation'])):
                    self.move_left()
            else:
                for _ in range(move['translation']):
                    self.move_right()
            self.moveAlgorithm = move['algorithm']
            
            self.output()
            
            # Update the score on the board
            self.update_score()
    
    def reset(self):
        '''Reset the grid for the next genome/generation or for a new game.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        # Clear everything
        self.grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.delete_grid()
        self.delete_score()
        self.score = 0
        self.movesTaken = 0
        self.new_storage()
        self.next_shape()
    
    def _random_genome(self):
        '''Returns a random genome from the genomes list of the tetris game.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        return self.genomes[random.randint(0, len(self.genomes) - 1)]
    

    def rotate_shape(self):
        '''Rotate the current game shape.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        self.remove_shape()
        self.currentShape['shape'] = _rotate(self.currentShape['shape'], 1)
        if self.collides(self.currentShape, self.grid):
            self.currentShape['shape'] = _rotate(self.currentShape['shape'], 3)
        self.apply_shape()
    
    def move_left(self):
        '''Moves the tetris piece left one block.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        self.remove_shape()
        self.currentShape['xCoord'] -= 1
        if self.collides(self.currentShape, self.grid):
            self.currentShape['xCoord'] += 1
        self.apply_shape()
    
    def move_right(self):
        '''Moves the tetris piece right one block.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        self.remove_shape()
        self.currentShape['xCoord'] += 1
        if self.collides(self.currentShape, self.grid):
            self.currentShape['xCoord'] -= 1
        self.apply_shape()
    
    def move_down(self):
        '''Moves the tetris piece down one block.
        
        Arguments:
            none
        
        Return Value:
            a tetris piece in its location representation after it has
            shifted downwards one block
        '''
        # Return a piece and its info after moving downwards
        result = {'lose': False,
                  'moved': True,
                  'rowsCleared': 0}

        self.remove_shape()
        self.currentShape['yCoord'] += 1
        
        if self.collides(self.currentShape, self.grid):
            # Hits the bottom of the grid
            self.currentShape['yCoord'] -= 1
            self.apply_shape()
            self.tempShape = self.currentShape.copy()
            self.next_shape()

            result['rowsCleared'] = self.clear_rows()
            # Hits the top of the grid as well
            if self.collides(self.currentShape, self.grid):
                result['lose'] = True
                if not self.ai:
                    self.reset()
            result['moved'] = False
            
        if result['moved'] == True:
            self.apply_shape()
            self.score += 1
            self.update_score()
        
        self.output()
        
        return result
    
    def clear_rows(self):
        '''Clears completed rows from the grid.
        
        Arguments:
            none
        
        Return Value:
            an integer representing the number of rows cleared from the grid.
        '''
        # Temporary fix for AI bug: self.grid randomly loses an empty row of 0s
        diff = len(self.grid) - 20
        if diff > 0:
            self.grid = self.grid[-20:]
        elif diff < 0:
            for _ in range(abs(diff)):
                output = [[0] * len(self.grid[0])]
                output.extend(self.grid)
                self.grid = output
                
        filledRows = []
        for row in range(len(self.grid)):
            notFilled = False
            for col in range(len(self.grid[row])):
                if self.grid[row][col] == 0:
                    notFilled = True
                    break
            if not notFilled:
                filledRows.append(row)
        
        # From traditional Tetris scoring rules
        if len(filledRows) == 1:
            self.score += 400
        elif len(filledRows) == 2:
            self.score += 1000
        elif len(filledRows) == 3:
            self.score += 3000
        elif len(filledRows) >= 4:
            self.score += 12000
        
        # Create a new grid with the filled rows removed
        output = []
        if self.testing == False:
            for row in filledRows:
                output.append([0] * len(self.grid[0]))
                self.grid.remove(self.grid[0])
        
        output.extend(self.grid)
        self.grid = output
        
        return len(filledRows)
        
    def remove_shape(self):
        '''Removes the shape from the board to clear the board.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        # Remove current shape and replace with a 0 (empty space)
        for row in range(len(self.currentShape['shape'])):
            for col in range(len(self.currentShape['shape'][row])):
                if self.currentShape['shape'][row][col] != 0:
                    self.grid[self.currentShape['yCoord'] + row]\
                             [self.currentShape['xCoord'] + col] = 0
    
    def apply_shape(self):
        '''Sticks the current shape onto the grid list. Does not automatically
        appear on the GUI. Transfers focus from the piece to the next piece.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        # Insert the value of the shape into the grid
        for row in range(len(self.currentShape['shape'])):
            for col in range(len(self.currentShape['shape'][row])):
                if self.currentShape['shape'][row][col] != 0:
                    self.grid[self.currentShape['yCoord'] + row]\
                             [self.currentShape['xCoord'] + col] = \
                             self.currentShape['shape'][row][col]
    
    def next_shape(self):
        '''Change to the next shape. Called after the apply_shape command.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        self.storageIndex += 1
        
        # If this is the first shape or the last shape of the storage list, we
        # generate a new storage list
        if len(self.storage) == 0 or self.storageIndex == len(self.storage):
            self.new_storage()
        # For the on deck shape display
        elif self.storageIndex == len(self.storage) - 1:
            # For debugging purposes
            self.onDeck = random.choice(self.storage)[1]
        else:
            self.onDeck = self.storage[self.storageIndex + 1][1]
        
        self.currentShape['shape'] = self.storage[self.storageIndex][1]
        self.currentShape['xCoord'] = ceil(GRID_WIDTH / 2)
        self.currentShape['yCoord'] = 0
    
    def new_storage(self):
        '''Creates a new storage list containing a permutation of the tetris
        pieces.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        newS = list(random.choice(list(permutate(list(self.shapes.items())))))
        
        self.storageIndex = 0
        self.storage = newS
        
    def get_state(self):
        '''Get current state of the game.
        
        Arguments:
            none
        
        Return Value:
            A dictionary that contains the current states (grid, currentShape,
            onDeck, storage, storageIndex, score) of the game
        '''
        state = {'grid': self.grid,
                 'currentShape': self.currentShape,
                 'onDeck': self.onDeck,
                 'storage': self.storage,
                 'storageIndex': self.storageIndex,
                 'score': self.score}
        
        return state
    
    def get_height(self):
        '''Gets the height of the highest column in the tetris grid.
        
        Arguments:
            none
        
        Return Value:
            an integer representing the height of the highest column in the
            tetris grid
        '''
        for row in range(len(self.grid)):
            for element in self.grid[row]:
                if element != 0:
                    return 20 - row
        
        return 0
    
    def get_cumulative_height(self):
        '''Gets the total sum of the heights of the columns in the tetris grid.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        transposedGrid = transpose(self.grid)
        height = 0
        
        for col in range(len(transposedGrid)):
            for row in range(len(transposedGrid[0])):
                if transposedGrid[col][row] != 0:
                    height += 20 - row
                    break
        
        return height
    
    def get_relative_height(self):
        '''Gets the absolute difference between the height of the highest
        column and the height of the lowest column in the tetris grid.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        highest = self.get_height()
        for row in range(len(self.grid) - 1, len(self.grid) - highest - 1, -1):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] == 0:
                    return highest - (19 - row)
        return 0
    
    def get_holes(self):
        '''Gets the number of holes in the tetris grid. A hole is an empty
        position on the tetris grid that has a filled position directly above.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        heights = [20] * len(self.grid[0])
        for col in range(len(self.grid[0])):
            for row in range(len(self.grid)):
                if self.grid[row][col] != 0:
                    if row < heights[col]:
                        heights[col] = row
        
        holes = 0
        for i in range(len(heights)):
            for row in range(heights[i] + 1, len(self.grid)):
                if self.grid[row][i] == 0:
                    holes += 1
        
        return holes
        
    def get_roughness(self):
        '''Gets the roughness of the tetris grid. The roughness is the sum of
        the absolute relative difference between the heights of each pair of
        adjacent columns.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        transposedGrid = transpose(self.grid)
        heights = [0] * GRID_WIDTH
        for col in range(GRID_WIDTH - 1):
            for row in range(GRID_HEIGHT):
                if transposedGrid[col][row] != 0:
                    heights[col] = 20 - row
                    break
        
        roughness = 0
        for i in range(len(heights) - 1):
            roughness += abs(heights[i + 1] - heights[i])
        
        return roughness
    
    def delete_score(self):
        '''Deletes the score so that self.update_score() can redraw the score.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        for tag in range(len(self.scoreID)):
            if self.scoreID[tag] != 0:
                self.canvas.delete(self.scoreID[tag])
                self.scoreID[tag] = 0
        
    def delete_grid(self):
        '''Deletes the grid so that self.output() can redraw the grid.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        '''
        for row in range(len(self.textID)):
            for col in range(len(self.textID[row])):
                try:
                    self.canvas.delete(self.textID[row][col])
                    self.textID[row][col] = 0
                except:
                    pass
        '''
        for row in range(len(self.textID)):
            for col in range(len(self.textID[row])):
                self.textID[row][col] = 0
        
        for value in self.values:
            self.canvas.delete(value)
        
    def output(self):
        '''Outputs the score to the screen. Planned to incorporate with a GUI.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        if self.draw and self.terminal == False:
            self.delete_grid()
            # Draw the grid
            for row in range(len(self.grid)):
                for col in range(len(self.grid[row])):
                    if self.grid[row][col] == 0:
                        t = self.canvas.create_text(50 + 20 * col, \
                                                    100 + 20 * row, \
                                                    fill='#111111', \
                                                    font=FONT, \
                                                    text=self.grid[row][col])
                    else:
                        t = self.canvas.create_text(50 + 20 * col, \
                                                    100 + 20 * row, \
                                    fill=self.colors[self.grid[row][col] - 1], \
                                    font=FONT, text=self.grid[row][col])
                    self.textID[row][col] = t
                    self.values.append(t)
            
            self.master.update()
        elif self.terminal == True and self.testing == False and not self.simp:
            for row in range(len(self.grid)):
                print('|', end='')
                for col in range(len(self.grid[row])):
                    print(self.grid[row][col], end='')
                print('|')
            
    def collides(self, shape, grid):
        '''Determines if object 1 collides with object 2.
        
        Arguments:
            shape: a dictionary (keys: xCoord, yCoord, shape (list of lists)
            that represents a tetris piece
            grid: the first object (the grid)
        
        Return Value:
            a boolean that represents whether the two objects collide
        '''
        for row in range(len(shape['shape'])):
            for col in range(len(shape['shape'][row])):
                if shape['shape'][row][col] != 0:
                    y = shape['yCoord'] + row
                    x = shape['xCoord'] + col
                    if y > len(self.grid) - 1:
                        return True
                    elif x < 0 or x > len(self.grid[0]) - 1:
                        return True
                    elif grid[y][x] != 0:
                        return True
        
        return False
    
    def play_tetris(self):
        '''Plays the actual tetris loop.
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        self.canvas.after(self.speed, self.start_loop())

# +------------------------------
# AI functions
# +------------------------------

    def create_initial_population(self):
        '''Creates the initial population and genomes for evolutionary
        genomics. Contains the following information:
            id: id tag [0, 1)
            weightedHeight: height of highest column ** 1.5 [-1, 1)
            rowsCleared: number of rows cleared [-1, 1)
            cumulativeHeight: sum of all heights [-1, 1)
            relativeHeight: range/difference between highest and lowest
                            columns [-1, 1)
            holes: number of holes [-1, 1)
            roughness: sum of absolute differences between adjacent columns
                       [-1, 1)
        
        Arguments:
            none
        
        Return Value:
            none
        '''
        # Creates the array
        self.genomes = []
        # genome will get the fitness characteristic later on
        for i in range(self.populationSize):
            genome = {'id': random.random(),
                      'rowsCleared': 2 * random.random() - 1,
                      'weightedHeight': 2 * random.random() - 1,
                      'cumulativeHeight': 2 * random.random() - 1,
                      'relativeHeight': 2 * random.random() - 1,
                      'holes': 2 * random.random() - 1,
                      'roughness': 2 * random.random() - 1}
            self.genomes.append(genome)
        self.evaluate_next_genome()
    
    def evaluate_next_genome(self):
        if self.simp:
            print(f'[Moves Taken: {self.movesTaken}, \
            Current Genome: {self.currentGenome + 1}, \
            Current Generation: {self.generation}, \
            Score: {self.score}]')            
        self.currentGenome += 1
        if self.currentGenome == len(self.genomes):
            self.evolve()
        self.reset()
        self.make_next_move()
    
    def evolve(self):
        '''Initiates the selection process for the evolutionary algorithm.'''
        print(f'Generation {self.generation}:')
        # Start the new genome and value of generation
        self.currentGenome = 0
        self.generation += 1
        
        # Start the GUI of the game over
        self.reset()
        
        # Select new generation of genomes using elites from past generation
        self.genomes.sort(key=lambda x: -x['fitness'])
        
        # ------------------------- SELECTION -----------------------------
        # Create collection of elite genomes
        for i in range(ELITES):
            self.archive['elites'].append(self.genomes[i])
            print(f'Elite {i + 1} has fitness {self.genomes[i]["fitness"]}')
            print(f'Elite {i + 1} has genome {self.genomes[i]}')
        
        # Get rid of the bottom 50% of genomes
        for _ in range(floor(len(self.genomes) / 2)):
            self.genomes.pop()
        
        totalFitness = 0
        for i in range(len(self.genomes)):
            totalFitness += self.genomes[i]['fitness']
        averageFitness = round(totalFitness / POPULATION_SIZE)
        self.averageFitness.append(averageFitness)
        print(f'Average fitness: {self.averageFitness}')
        
        # Create an array of children genomes
        children = []
        iterations = min(ELITES, len(self.genomes))
        for elite in range(iterations):
            children.append(self.genomes[elite])
        
        while len(children) < POPULATION_SIZE:
            children.append(self.make_child(self.genomes[0], self.genomes[0]))
        
        # The bottom two don't do anything
        while len(children) < POPULATION_SIZE / 2:
            children.append(self.make_child(self._random_genome(), \
                            self.genomes[random.randint(0, iterations - 1)]))
        while len(children) < POPULATION_SIZE:
            children.append(self.make_child(\
                self.genomes[random.randint(0, iterations - 1)], \
                self.genomes[random.randint(0, iterations - 1)]))
        # Update current genomes to children genomes
        self.genomes = []
        self.genomes.extend(children)
        
        # Store info database
        self.archive['genomes'] = self.genomes.copy()
        self.archive['currentGeneration'] = self.generation
        
    def make_child(self, parent1, parent2):
        '''Creates a child genome from the data of two parent genomes.'''
        
        # ------------------------- CROSSOVER -----------------------------
        child = {'id': random.random(),
                 'rowsCleared': _make_child(parent1, parent2, 'rowsCleared'),
                 'weightedHeight': _make_child(parent1, parent2, \
                                               'weightedHeight'),
                 'cumulativeHeight': _make_child(parent1, parent2, \
                                               'cumulativeHeight'),
                 'relativeHeight': _make_child(parent1, parent2, \
                                               'relativeHeight'),
                 'holes': _make_child(parent1, parent2, 'holes'),
                 'roughness': _make_child(parent1, parent2, 'roughness'),
                 'fitness': -1}
        
        # ------------------------- MUTATION ------------------------------
        # We want a mutation range of [-mutationStep, mutationStep)
        # 2 * (random from [0, 1)) - 1 gives [-1, 1)
        
        childKeys = list(child.keys())
        
        for i in range(1, 7):
            if random.random() < MUTATION_RATE:
                child[childKeys[i]] += (2 * random.random() - 1) * MUTATION_STEP
        
        return child
    
    def possible_moves(self):
        '''Creates a list of possible moves and the ratings of the moves.
        
        Arguments:
            none
        
        Return Value:
            moves: a list of all possible moves
        '''
        lastState = self.get_state()
        moves = []
        
        self.testing = True
        # Tries every possible move and their ratings
        for rotations in range(4):
            oldXCoords = []
            
            # Possible horizontal shift locations
            for xShift in range(-ceil(GRID_WIDTH / 2) - 1, \
                                floor(GRID_WIDTH / 2)):
                self.load_state(lastState)
                
                # Possible rotations
                for _ in range(rotations):
                    self.rotate_shape()
                
                if xShift < 0:
                    for _ in range(abs(xShift)):
                        self.move_left()
                else:
                    for _ in range(xShift):
                        self.move_right()
                
                # This checks whether the current rotation has already been
                # accounted for (e.g. square after rotation is the same square)
                if self.currentShape['xCoord'] not in oldXCoords:
                    
                    # Move the block down as far as possible
                    moveDown = self.move_down()
                    while(moveDown['moved']):
                        moveDown = self.move_down()
                    
                    algorithm={'rowsCleared': moveDown['rowsCleared'],
                               'weightedHeight': pow(self.get_height(), 1.5),
                               'cumulativeHeight': self.get_cumulative_height(),
                               'relativeHeight': self.get_relative_height(),
                               'holes': self.get_holes(),
                               'roughness': self.get_roughness()}
                    
                    for key in list(algorithm.keys()):
                        algorithm[key] *= (1 + random.choice([RANDOM_MUTATION, \
                                                             -RANDOM_MUTATION]))
                    
                    rating = 0
                    
                    genomeKeys = list(self.genomes[self.currentGenome].keys())
                    for i in range(1, 7):
                        rating += algorithm[genomeKeys[i]] * \
                            self.genomes[self.currentGenome][genomeKeys[i]]
                    
                    if moveDown['lose']:
                        rating -= LOSS_VALUE
                    
                    moves.append({'rotation': rotations,
                                  'translation': xShift,
                                  'rating': rating,
                                  'algorithm': algorithm})
                    oldXCoords.append(self.currentShape)
                    
                    self.temp2Shape = self.currentShape.copy()
                    self.currentShape = self.tempShape.copy()
                    self.remove_shape()
                    self.currentShape = self.temp2Shape.copy()
        
        self.testing = False
        self.load_state(lastState)
        
        return moves
    
    def get_best_move(self, moves):
        '''Returns the best/highest rated move from the inputted set of moves.
        
        Arguments:
            moves: a list of moves
        
        Return Value:
            a move that has the highest rating out of the list of inputted moves
        '''
        bestRating = -4e9
        bestMove = [moves[0]]
        
        for move in moves:
            if move['rating'] > bestRating:
                bestRating = move['rating']
                bestMove = [move]
            elif move['rating'] == bestRating:
                bestMove.append(move)
        
        return random.choice(bestMove)
    
    def play_ai(self):
        '''Starts the training with ai.'''
        self.archive['populationSize'] = self.populationSize
        self.next_shape()
        self.apply_shape()
        self.saveState = self.get_state()
        self.roundState = self.get_state()
        
        self.create_initial_population();
        self.start_loop()
        
if __name__ == '__main__':
    root = Tk()
    tetris = Tetris(root)
    tetris.play_ai()
    
    while not tetris.done:
        tetris.play_tetris()
