
# Tetris AI in Python

Version 1 of a Tetris AI written in Python

The AI utilizes an evolutionary algorithm that uses reinforcement learning to select well-performing rulesets within each generation to be a model/reference point in the creation of rulesets in the subsequent generation. Eventually, the AI will reach a local minimum ruleset that will allow it to autonomously play the game while efficiently scoring.

How to use:

In the __init__ constructor of the Tetris class, change the following values to preference:

    self.ai: set to True if you want AI to play
    self.terminal: set to True if you want output to Terminal. Otherwise, a tkinter canvas will be created (slows down dramatically as item handles add up --> not suitable for AI)
    self.simp: set to True if simple outputs are desired. Otherwise, it will output information every time the score/move counter updates
    self.speed: set to a non-negative integer that represents the delay in milliseconds between updates of the game state

Then run the TetrisAI.py file from terminal or an IDE and enjoy!

What will be added in the future:
* Implementing parallelization by splitting the breadth-first search to different CUDA cores in a GPU. Will decrease computation time by 20x (estimated)
* Adding a normalization factor to the weights in the algorithm/genome (currently, they tend to favor values in the range of 1e-9 and 1e-12)
* Separating the TetrisAI.py file into smaller, more manageable files
* Fixing the significant speed drop due to using too many tkinter.Canvas item handles
* More data visualization during training (e.g. create a plot of the average fitness of each generation, correlation between specific paramters and fitness)
* Output to CSV/other manageable file formats

Keybinds:
* Down arrow: move block down
* Up arrow: rotate block 90 degrees clockwise
* Left arrow: move block left
* Right arrow: move block right
* a/A: turn on/off AI
* u/U: speed up the game
* i/I: slow down the game
* s/S: save current state of the game (for debugging purposes)
* l/L: load saved state of the game (for debugging purposes)
* q/Q: quit game
