# search.py
# ---------
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

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import util
import heapq

class SearchProblem:

    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem): # Returns solution or failure
    
    node = {"state": problem.getStartState(), "cost": 0 } # A node is presented as his state and path cost
    
    if problem.isGoalState(node["state"]): # In case we start where we want to finish
        return []
    
    frontier = util.Stack() # The frontier is a queue
    explored = set() # The nodes we explored already
    frontier.push(node) # Adds the node to the frontier
    
    while True:
        if frontier.isEmpty():
            raise Exception("failure")
        
        node = frontier.pop() # Always choose the deepest node
        
        if problem.isGoalState(node["state"]):
            actions = []
            while "parent" in node:
                actions.append(node["action"])
                node = node["parent"]
            actions.reverse()
            return actions # returns the path made to reach goal node

        explored.add(node["state"])
        
        successors = problem.getSuccessors(node["state"])
        for successor in successors:
            # Successor's format: (state, action, cost)
            child = {"state": successor[0], "action": successor[1], \
                     "cost": node["cost"] + successor[2], "parent": node}
            if child["state"] not in explored:
                frontier.push(child)
                                      
def breadthFirstSearch(problem): # Returns solution or failure
    """Search the shallowest nodes in the search tree first."""
    node = {"state": problem.getStartState(), "cost": 0 } # A node is presented as his state and path cost
    
    if problem.isGoalState(node["state"]): # In case we start where we want to finish
        return []
    
    frontier = util.Queue() # The frontier is a queue
    explored = set() # The nodes we explored already
    frontier.push(node) # Adds the node to the frontier
    
    while True:
        if frontier.isEmpty():
            raise Exception("failure")
        
        node = frontier.pop() # Choose the most shallow node in frontier
        
        if problem.isGoalState(node["state"]):
            actions = []
            while "parent" in node:
                actions.append(node["action"])
                node = node["parent"]
            actions.reverse()
            return actions # returns the path made to reach goal node

        explored.add(node["state"])
        
        successors = problem.getSuccessors(node["state"])
        for successor in successors:
            # Successor's format: (state, action, cost)
            child = {"state": successor[0], "action": successor[1], \
                     "cost": node["cost"] + successor[2], "parent": node}
            if child["state"] not in explored:
                frontier.push(child)
                explored.add(child["state"])
   
def update(frontier,item, priority):
    # If item already in priority queue with higher priority, update its priority and rebuild the heap.
    # If item already in priority queue with equal or lower priority, do nothing.
    for index, (p, c, i) in enumerate(frontier.heap):
        if i["state"] == item["state"]:
            if p <= priority:
                break
            del frontier.heap[index]
            frontier.heap.append((priority, c, item))
            heapq.heapify(frontier.heap)
            break
    else:
        # Item not in priority queue, do the same thing as self.push.
        frontier.push(item, priority)
   
def uniformCostSearch(problem): # Returns solution or failure
    """Search the node of least total cost first."""
     
    node = {"state": problem.getStartState(), "cost": 0} # Node format: state and cost
    
    if problem.isGoalState(node["state"]): # In case we start where we want to finish
        return []
    
    frontier = util.PriorityQueue() # The frontier is a priority queue
    frontier.push(node, node["cost"]) # Adds the node to the frontier
    explored = set()# The nodes we explored already
    
    while True:
        if frontier.isEmpty():
            raise Exception("failure")
          
        node = frontier.pop()# Choose the lowest cost node in frontier
             
        if problem.isGoalState(node["state"]): 
            actions = []
            while "parent" in node:
                actions.append(node["action"])
                node = node["parent"] 
            actions.reverse()
            return actions # returns the path made to reach goal node
               
        explored.add(node["state"]) 
         
        successors = problem.getSuccessors(node["state"])
        for successor in successors:
            # Successor's format: (state, action, cost)
            child = {"state": successor[0], "action": successor[1], \
                     "cost": node["cost"] + successor[2], "parent": node}
            if child["state"] not in explored:
                update(frontier, child, child["cost"])
          
             
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    
    node = {"state" : problem.getStartState() ,"cost" : 0} # Node format: state and cost
    
    if problem.isGoalState(node["state"]):# In case we start where we want to finish
        return []

    frontier = util.PriorityQueue() # The frontier is a priority queue
    # Each node's priority of the queue is calculated by the node's cost plus the heuristic function given
    frontier.push(node, node["cost"] + heuristic(node["state"], problem))

    explored = set() # The nodes we explored already

    while True:
        if frontier.isEmpty():
            raise Exception("failure")
        node = frontier.pop()
       
        if problem.isGoalState(node["state"]):
            actions = []
            while "parent" in node:
                actions.append(node["action"])
                node = node["parent"]
            actions.reverse()
            return actions # returns the path made to reach goal node
       
        explored.add(node["state"])
        
        successors = problem.getSuccessors(node["state"])
        for successor in successors:
            child = {"state": successor[0], "action":successor[1],
                     "cost":successor[2] + node["cost"], "parent":node}
            if child['state'] not in explored:
                update(frontier, child, child["cost"] + heuristic(successor[0], problem))

# Abbreviations

bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
