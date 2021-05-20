# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

# Question 1
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]
 
    def getNextPositions(self, walls, pos):
        # Return all next positions the agent can move to from 
        # his position without hitting a wall
        for row_step, col_step in [(1,0), (-1,0), (0,1), (0,-1)]:
            row = pos[0] + row_step
            col = pos[1] + col_step
            if not walls[row][col]:
                yield row, col
    
    # BFS algorithm
    def minMazeDistance(self, walls, pos1, positions):
        # Return the walk distance from "pos1"(agent's position) to any of "positions"
        # (according to request - food or capsules) without hitting walls
        
        # If agent's position is on any one of the requested positions - return 0 
        if pos1 in positions:
            return 0
        
        # Our frontier is a queue for bfs implementation
        frontier = util.Queue()
        # Adds the position to the frontier
        frontier.push((pos1, 0))
        # The positions we visited already
        visited = set([pos1]) # Make a set of one element
        
        while not frontier.isEmpty():
            # Choose the most shallow agent's position in the frontier
            (pos, depth) = frontier.pop() 
            for new_pos in self.getNextPositions(walls, pos):
                if new_pos in positions:
                    # Return the depth of the first element in position list
                    return depth + 1
                # If new position already in visited do nothing.
                elif new_pos in visited:
                    continue
                # Adds the new position to the frontier
                frontier.push((new_pos, depth + 1))
                # Adds the new position to visited
                visited.add(new_pos)
        # Means none of the positions got found
        return walls.height * walls.width 
    
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        
        
        # The agent's state successor  
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # The agent's new position
        newPos = successorGameState.getPacmanPosition()
        # A list of all food that are currently on the maze
        oldFood = currentGameState.getFood().asList()
        # A list of all capsules currently on the maze
        oldCapsules = currentGameState.getCapsules()
        # A string that shows the walls of the maze
        walls = currentGameState.getWalls()
        # Size of our maze
        gridSize = walls.height * walls.width
        # A list of ghosts's state
        newGhostStates = successorGameState.getGhostStates()
        # An evaluation for each agent's action: price for good action or fine for bad action
        evaluation = successorGameState.getScore()
        
        
        # Be careful from ghosts 
        for ghostState in newGhostStates:
            # Ghost position on the maze
            ghostPos = ghostState.getPosition()
            # If ghost is close - get a fine
            if util.manhattanDistance(ghostPos, newPos) < 2:
                evaluation -= gridSize
        # Reach for a capsule is a price
        # Smaller distance to food\capsule means higher score
        if oldCapsules:
            evaluation -= 2 * self.minMazeDistance(walls, newPos, oldCapsules)
        # Reach for food is a price
        if oldFood:
            evaluation -= self.minMazeDistance(walls, newPos, oldFood)
        
        # Return evaluation summation calculated for each agent's action
        return evaluation
      
  
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

# Question 2
class MinimaxAgent(MultiAgentSearchAgent):
    
    def minimax(self, gameState, depth, index):
        # Initialize action to unknown(None)
        action = "None"
        
        # Terminal Test: in case of winning or losing or >= depth given 
        if gameState.isWin() or gameState.isLose() or depth >= self.depth:        
            v = self.evaluationFunction(gameState)
            return v, action
        
        # Initialize value for pacman agent
        if index == 0:
            value = -float("inf")
        # Initialize value for a ghost agent
        else:
            value = float("inf") 
        
        numOfAgents = gameState.getNumAgents()    
        legalActions = gameState.getLegalActions(index) 
        
        for move in legalActions:
            successor = gameState.generateSuccessor(index, move)
            # Get next agent index
            newIndex = (index + 1) % numOfAgents
            # Get next agent depth
            newDepth = depth
            # If next agent is pacman increase depth by 1
            if newIndex == 0:
                newDepth += 1
            next = self.minimax(successor, newDepth, newIndex)
            newValue = next[0]
            newAction = next[1]
            # Do min or max depends on agent type
            if (index != 0 and value > newValue) or (index == 0 and value < newValue):
                value = newValue
                action = move
        return value, action

    # Returns the minimax action from the current gameState
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(index):
            Returns a list of legal a for an agent
            index=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(index, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        result = self.minimax(gameState, 0, 0)
        return result[1]   
      
# Question 3
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabeta(self, alpha, beta, gameState, depth, index):
        
        # Initialize action to unknown(None)
        action = "None"
        
        # Terminal Test: in case of winning or losing or >= depth given 
        if gameState.isWin() or gameState.isLose() or depth >= self.depth:        
            v = self.evaluationFunction(gameState)
            return v, action
        
         # Initialize value for pacman agent
        if index == 0:
            value = -float("inf")
        # Initialize value for a ghost agent
        else:
            value = float("inf") 
        
        numOfAgents = gameState.getNumAgents()    
        legalActions = gameState.getLegalActions(index)    
            
        for move in legalActions:
            successor = gameState.generateSuccessor(index, move)
            # Get next agent index 
            newIndex = (index + 1) % numOfAgents
            # Get next agent depth
            newDepth = depth
            # If next agent is pacman increase depth by 1
            if newIndex == 0:
                newDepth += 1  
            next = self.alphabeta(alpha, beta, successor, newDepth, newIndex)
            newValue = next[0]
            newAction = next[1]
            # Check if current agent is a ghost
            if index != 0 and value > newValue:   
                value = newValue
                action = move
                # Update beta and prune
                if value <= beta:    
                    beta = value
                if value < alpha:
                    return value, action
            # Check if current agent is pacman
            if index == 0 and value < newValue: 
                value = newValue
                action =  move
                # Update alpha and prune
                if value >= alpha:    
                    alpha = value
                if value > beta:
                    return value, action
        
        return value, action

    # Returns the minimax action from the current gameState 
    def getAction(self, gameState):
        result = self.alphabeta(-float("inf"), float("inf"), gameState, 0, 0)
        return result[1]
    
# Question 4
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
        Your expectimax agent (question 4)
    """
    # Returns the action of the expectiMax result
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # A root node will have a depth of 0
        depth = 0 
        # The pacman's index is always 0
        agentIndex = 0
        # Get the expectiMax function result 
        result = self.expValue(gameState, depth, agentIndex)
        # Return the expectiMax action
        return result[1] 
    
    
    # Returns a tuple: 
    # Value - the expectimax value
    # The action that lead to the expectimax value
    def expValue(self, gameState, depth, index):
        # Initialize action to unknown(None)
        action = "None"
        numOfAgents = gameState.getNumAgents()
        # Terminal Test: in case of winning or losing or >= depth given
        if gameState.isWin() or gameState.isLose() or depth >= self.depth:  
            v = self.evaluationFunction(gameState)
            return v, action
        legalActions = gameState.getLegalActions(index)
        actionsNum = len(legalActions)
        
        if index == 0:
            # Initialize value for pacman agent
            value = -float("inf") 
        else:
            # Initialize value for ghost agent
            value = 0  
            
        for move in legalActions:
            successor = gameState.generateSuccessor(index, move)
            # Get next agent index 
            newIndex = (index + 1) % numOfAgents
            # Get next agent depth 
            newDepth = depth
            # If next agent is pacman increase depth by 1
            if newIndex == 0:
                newDepth += 1 
            next = self.expValue(successor, newDepth, newIndex)
            newValue = next[0]
            newAction = next[1]
            # Check if current agent is a ghost
            # Calculate the average of node's children values and
            # add  average to value
            if index != 0:  
                avg = float(newValue) / float(actionsNum)
                value += avg
            # Check if current agent is pacman
            if index == 0 and value < newValue:  
                value = newValue
                action = move

        return value, action

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

