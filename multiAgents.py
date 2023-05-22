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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide. You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState) -> str:
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
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

        # calculate reciprocal of distances to food and ghosts
        foodDistances = [1.0 / util.manhattanDistance(newPos, food) for food in newFood.asList()]
        ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # Calculate score based on the distances to food and ghosts
        score = successorGameState.getScore()
        if ghostDistances and min(ghostDistances) <= 1 and not any(newScaredTimes):
            score -= 99999999
        else:
            score += sum(foodDistances) - min(ghostDistances)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def max_value(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, min_value(successor, depth, agentIndex + 1))
            return v

        def min_value(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex < gameState.getNumAgents() - 1:
                    v = min(v, min_value(successor, depth, agentIndex + 1))
                else:
                    v = min(v, max_value(successor, depth + 1, 0))
            return v

        def betterEvaluationFunction(gameState):
            """
                Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
                evaluation function (question 5).

                DESCRIPTION: Give high score to state with many food dots, few ghosts and
                close distance to the closest food dot. Penalize for number of capsules left.

            """

        food = gameState.getFood().asList()
        pacmanPos = gameState.getPacmanPosition()
        ghostStates = gameState.getGhostStates()

        # Score for number of food dots
        foodScore = len(food)

        # Score for distance to closest food dot
        if len(food) > 0:
            minDistance = min([manhattanDistance(pacmanPos, foodPos) for foodPos in food])
        else:
            minDistance = 0

        # Score for number of ghosts
        numGhosts = len(ghostStates)
        ghostScore = sum([ghostState.scaredTimer == 0 for ghostState in ghostStates])

        # Score for number of capsules left
        numCapsulesLeft = len(gameState.getCapsules())

        return foodScore - minDistance - numGhosts - numCapsulesLeft



    def getAction(self, gameState):
        
        def max_value(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, min_value(successor, depth, agentIndex + 1))
            return v

        def min_value(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex < gameState.getNumAgents() - 1:
                    v = min(v, min_value(successor, depth, agentIndex + 1))
                else:
                    v = min(v, max_value(successor, depth + 1, 0))
            return v

        legalMoves = gameState.getLegalActions(0)
        scores = [min_value(gameState.generateSuccessor(0, action), 0, 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            v = -float("inf")
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, agent, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float("inf")
            for action in state.getLegalActions(agent):
                if agent == state.getNumAgents() - 1:
                    v = min(v, max_value(state.generateSuccessor(agent, action), depth + 1, alpha, beta))
                else:
                    v = min(v, min_value(state.generateSuccessor(agent, action), depth, agent + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        alpha = -float("inf")
        beta = float("inf")
        v = -float("inf")
        for action in legalActions:
            next_state = gameState.generateSuccessor(0, action)
            next_v = min_value(next_state, 0, 1, alpha, beta)
            if next_v > v:
                v = next_v
                bestAction = action
            alpha = max(alpha, v)
        return bestAction


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
        # define expectimax function
        def expectimax(state, agentIndex, depth):
            # check if terminal state or maximum depth has been reached
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            # if agent is pacman, choose action with maximum value
            if agentIndex == 0:
                value = -float("inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, agentIndex+1, depth))
                return value
            # if agent is a ghost, average values of all legal actions
            else:
                value = 0
                actions = state.getLegalActions(agentIndex)
                # equal probability of all actions
                p = 1.0/len(actions)
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    # if there are still ghosts, evaluate next ghost
                    if agentIndex < state.getNumAgents() - 1:
                        value += expectimax(successor, agentIndex+1, depth)
                    # if all ghosts have been evaluated, evaluate pacman
                    else:
                        value += expectimax(successor, 0, depth-1)
                return value * p
        
        # choose action with maximum value
        bestAction = None
        bestValue = -float("inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 1, self.depth)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
    
    