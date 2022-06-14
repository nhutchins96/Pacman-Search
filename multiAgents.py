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


from cgitb import small
from cmath import inf
from dis import dis
from re import I
from tkinter.messagebox import NO
from turtle import pos
from util import manhattanDistance, pause
from game import Directions
import random, util


from game import Agent

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = 0
        
        # GHOST INFORMATION (it's a list of ghost agents) 
        # Find closest ghost
        ghosts_near_me = []
        for ghost in newGhostStates:
          ghosts_near_me.append(manhattanDistance(newPos, ghost.getPosition()))

        # The closer the ghost, the more impact on how bad the score is
        closest_ghost = min(ghosts_near_me)
                   
        # FIND FOODS
        closest_food = self.find_closest_food(newFood.asList(), newPos)
        
        # Overwrite closest food if we are within 2 blocks of capsules
        caps_near_me = []
        for cap in successorGameState.getCapsules():
          caps_near_me.append(manhattanDistance(newPos, cap))
        
        # This will encourage pacman to walk closer when within
        if len(caps_near_me):
          smallest_cap = min(caps_near_me)
          if smallest_cap <= 2:
            smallest_cap = 20
          else:
            smallest_cap = 0
        else:
          smallest_cap = 0
        
        # Bad to stand still because of timer
        if action == 'Stop':
          score -= 100
        
        # Determine if we can eat ghost in time
        eating_ghosts = 0
        if newScaredTimes[0]:
          # Can catch ghost if distance is half the amount of time
          time_left = newScaredTimes[0]
          if time_left >= 3*closest_ghost:
            eating_ghosts = 50
        
        return successorGameState.getScore() + closest_ghost / (closest_food*10) + score - len(caps_near_me)*30 + smallest_cap + eating_ghosts
    
    def find_closest_food(self, newFood, currentPos):
      distance = []
      
      # Loop through all food
      for food in newFood:
        distance.append(manhattanDistance(currentPos, food))
        
      # 0 if nothing here
      if len(newFood) == 0:
        distance = [1]
          
      return min(distance)


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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        """ 
        Player index = 0 which is when we want to MAX
        When index != 0 we are ghost and want to min
        """
        # Return the action from the best move we selected
        return self.minimax(gameState, 0, 0)[0]  

    def minimax(self, gameState, depth_counter, agent_index):
        # If agent index is over ghost count then we have completed one depth cycle
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            depth_counter += 1
            
        # Reached the depth limit
        if depth_counter == self.depth:
            return None, self.evaluationFunction(gameState)
          
        # This is to store the best score we find and best move (either min or max this way)
        best_score = None
        best_action = None
        
        # Loop through each possible action finding the minimax scores
        for action in gameState.getLegalActions(agent_index):
          # Get next game state and the minimax score
          successor_state = gameState.generateSuccessor(agent_index, action)
          score = self.minimax(successor_state, depth_counter, agent_index + 1)[1]
        
          # Pacman (MAX)
          if agent_index == 0: 
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
                
          # Ghost (MIN)
          else:
            if best_score is None or score < best_score:
                best_score = score
                best_action = action
                    
        # No successor states found aka a leaf
        if best_score is None:
            best_score = self.evaluationFunction(gameState)
          
        return best_action, best_score
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Now call with alpha and beta values used for pruning!
        return self.minimax(gameState, 0, 0, -inf, inf)[0]  

    def minimax(self, gameState, depth_counter, agent_index, alpha, beta):
        # If agent index is over ghost count then we have completed one depth cycle
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            depth_counter += 1
            
        # Reached the depth limit
        if depth_counter == self.depth:
            return None, self.evaluationFunction(gameState)
          
        # This is to store the best score we find and best move (either min or max this way)
        best_score = None
        best_action = None
        
        # Loop through each possible action finding the minimax scores
        for action in gameState.getLegalActions(agent_index):
          # Get next game state and the minimax score
          successor_state = gameState.generateSuccessor(agent_index, action)
          score = self.minimax(successor_state, depth_counter, agent_index + 1, alpha, beta)[1]
        
          # Pacman (MAX)
          if agent_index == 0: 
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
          
            # Pruning
            if best_score > beta:
              return best_action, best_score
            
            alpha = max(alpha, best_score)
                
          # Ghost (MIN)
          else:
            if best_score is None or score < best_score:
              best_score = score
              best_action = action
            
            # Pruning
            if best_score < alpha:
              return best_action, best_score 
            
            beta = min(beta, best_score)
                    
        # No successor states found aka a leaf
        if best_score is None:
            best_score = self.evaluationFunction(gameState)
          
        return best_action, best_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, 0)[0] 

    def minimax(self, gameState, depth_counter, agent_index):
        # If agent index is over ghost count then we have completed one depth cycle
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            depth_counter += 1
            
        # Reached the depth limit
        if depth_counter == self.depth:
            return None, self.evaluationFunction(gameState)
          
        # This is to store the best score we find and best move (either min or max this way)
        best_score = None
        best_action = None
        
        # Loop through each possible action finding the minimax scores
        for action in gameState.getLegalActions(agent_index):
          # Get next game state and the minimax score
          successor_state = gameState.generateSuccessor(agent_index, action)
          score = self.minimax(successor_state, depth_counter, agent_index + 1)[1]
        
          # Pacman (MAX)
          if agent_index == 0: 
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
                
          # Ghost (MIN) expectimax
          else:
            # Turn the score into probability 1/number of ghost actions
            prob = 1.0 / len(gameState.getLegalActions(agent_index))
            
            # Change best_score from none if first iter
            if best_score == None:
              best_score = 0.0
              
            best_score += score * prob
            best_action = action

                    
        # No successor states found aka a leaf
        if best_score is None:
            best_score = self.evaluationFunction(gameState)
          
        return best_action, best_score
      
      
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I found the values for everything that was relevant to the score.
      The food distance - the smaller the better 
      The food amount - the less the better 
      The caps amount - the less the better
      The ghost distance - the smaller the worse
      The ghost eating count - step on white ghost is good
      If the move results in a win or lose
      
      Once I found all the values, I played around with weighting them until I found a combination that worked.
      The most important was making sure pacman prioritized eating food.
      I had to make the distance not very impactful otherwise pacman would prefer to keep a distance from food.
    """
    current_pos = currentGameState.getPacmanPosition()
    
    # Find closest food pellet
    food = currentGameState.getFood().asList()
    closest_food_distance = find_closest_food(food, current_pos)
    
    # Find closest ghost
    newGhostStates = currentGameState.getGhostStates()
    ghosts_near_me = []
    for ghost in newGhostStates:
        ghosts_near_me.append(manhattanDistance(current_pos, ghost.getPosition()))

    # The closer the ghost, the more impact on how bad the score is
    closest_ghost_distance = min(ghosts_near_me)
    
    # The smaller these are the better
    remaining_food = currentGameState.getNumFood()
    remaining_caps = len(currentGameState.getCapsules())

    # Check if this is a win or lose state
    win_lose = 0
    if currentGameState.isWin():
      win_lose = inf
    if currentGameState.isLose():
      win_lose = -1000
      
    # Determine if we can eat ghost in time
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    eating_ghosts = 0
    if newScaredTimes[0]:
      # Can catch ghost if distance is close by
      time_left = newScaredTimes[0]
      if time_left >= 3*closest_ghost_distance:
        eating_ghosts = 1
    
    ### Weight values! ###
    # We can eat the ghost if it's scared 
    if closest_ghost_distance == 0 and eating_ghosts == 1:
      eating_ghosts = 5000
    
    # The smaller the distance to food the better! e.g. 1 block away will be 50, 3 blocks will be 25
    food_distance_weighted = 1/(closest_food_distance + 1) * 100
  
    # The less food in the game the better
    food_count_weighted = 1/(remaining_food + 1) * 50000
    
    # The less capsules the better 
    caps_count_weighted = 1/(remaining_caps + 1) * 5000
    
    # The closer the ghost the worse the score!
    ghost_distance_weighted = 0
    if closest_ghost_distance < 2:
      ghost_distance_weighted = -1000
    
    return food_distance_weighted + food_count_weighted + caps_count_weighted \
        + ghost_distance_weighted + win_lose + eating_ghosts

def find_closest_food(newFood, currentPos):
    """
      Used in the previous eval function. This finds the closest food item
    """
    distance = []

    # Loop through all food
    for food in newFood:
      distance.append(manhattanDistance(currentPos, food))
      
    # 0 if nothing here
    if len(newFood) == 0:
      distance = [1]
        
    return min(distance)   
    
# Abbreviation
better = betterEvaluationFunction

