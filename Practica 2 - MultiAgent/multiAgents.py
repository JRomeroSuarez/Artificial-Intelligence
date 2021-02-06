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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        food_list = newFood.asList()
        currentScore = scoreEvaluationFunction(currentGameState)
        food_coordinates = []
        scared_ghost = []
        bad_ghost = []
        minscared = 1000
        minbad = 1000

        if successorGameState.isWin():  # Si el següent estat es guanyador retorna el score maxim
            return 999999999
        for foods in food_list:  # Iterem per la llista de foods
            food_coordinates.append(manhattanDistance(newPos, foods))  # Afegim les posicions del menjar mes proxim

        closest_food = min(food_coordinates)  # Agafem el minim dels menjars
        for ghost_states in newGhostStates:  # Per cada estat del fantasmes
            if newPos == ghost_states.getPosition():  # Comprovem que si colisionen perdin
                return -1
            else:
                if ghost_states.scaredTimer:
                    scared_ghost.append(
                        manhattanDistance(newPos, ghost_states.getPosition()))  # Afegim els scared ghosts
                else:
                    bad_ghost.append(manhattanDistance(newPos, ghost_states.getPosition()))  # Afegim els bad ghosts

        if len(scared_ghost):  # Si encara queden scared ghosts
            minscared = min(scared_ghost)  # Agafem la minima distancia d'aquest
        if len(bad_ghost):  # Si encara queden bad_ghosts
            minbad = min(bad_ghost)  # Agafem la minima distancia d'aquest
        score = (1.0 / closest_food) + (1.0 / minscared) - (
                1.0 / minbad)  # Calculem el score amb els values
        return score + successorGameState.getScore()  # Retornem el score i el score del succesor


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    # Funcio del maxim valor utilitzada unicament pel calcul del pacman
    def maxValue(self, gameState, depth, agents):
        if gameState.isWin() or gameState.isLose() or depth == 0:  # base cases
            return self.evaluationFunction(gameState)

        score = -999999999

        for legalActions in gameState.getLegalActions(0):  # Per cada accio permesa
            successor = gameState.generateSuccessor(0, legalActions)  # Generem els succesors pel pacman
            score = max(score, self.minValue(successor, depth, 1,
                                             agents))  # Mantenim el maxim entre els valors minims del succesors

        return score

    # Funcio de valor minim
    def minValue(self, gamestate, depth, agents, ghosts):
        if gamestate.isWin() or gamestate.isLose() or depth == 0:  # Cas base
            return self.evaluationFunction(gamestate)

        score = 999999999
        for legalActions in gamestate.getLegalActions(agents):  # Per cada accio permesa
            succesors = gamestate.generateSuccessor(agents, legalActions)  # Generem els succesors
            if agents == ghosts:
                score = min(score, self.maxValue(succesors, depth - 1,
                                                 ghosts))  # Mantenim el minim entre els valors maxims del succesors
            else:  # En cas que siguin altres ghosts
                score = min(score, self.minValue(succesors, depth, agents + 1,
                                                 ghosts))  # Mantenim el minim entre els valors minims del succesors
        return score

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

        ghosts = gameState.getNumAgents() - 1
        maxactions = []
        score = -999999999  # Init del score

        for legalActions in gameState.getLegalActions():  # Per cada accio permesa
            succesors = gameState.generateSuccessor(0, legalActions)  # Generem els succesors
            temp = score
            score = max(score, self.minValue(succesors, self.depth, 1,
                                             ghosts))  # Comprovem el maxim entre els succesors que tenen valors minims
            if score > temp:  # Guardem el score maxim
                maxactions.append(legalActions)

        while len(maxactions) != 0:  # Finalment retornem els maxim valors
            return maxactions.pop()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # Funcio del maxim valor utilitzada unicament pel calcul del pacman
    def max_value(self, gameState, depth, ghosts, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:  # Cas base
            return self.evaluationFunction(gameState)

        v = -999999999  # Score inicial

        for legalActions in gameState.getLegalActions(0):  # Iterem per les accions del pacman
            succesors = gameState.generateSuccessor(0, legalActions)  # Generem els succesors de les accions
            v = max(v, self.min_value(succesors, depth, 1, ghosts, alpha,
                                      beta))  # Comprovem el maxim entre els succesors que tenen valors minims

            # Comprovem que el score sigui mayor a beta, en cas contrari farem el maxim entre alpha y score per la
            # seguent iteracio
            if v > beta:
                return v
            alpha = max(alpha, v)

        return v

    # Funcio de valor minim
    def min_value(self, gameState, depth, agents, ghosts, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:  # Cas base
            return self.evaluationFunction(gameState)

        v = 999999999  # Score inicial
        for legalActions in gameState.getLegalActions(agents):  # Iterem per les accions del ghost
            succesors = gameState.generateSuccessor(agents, legalActions)  # Generem els succesors de les accions
            if agents == ghosts:  # Comprovem que sigui torn del ghost
                v = min(v, self.max_value(succesors, depth - 1, ghosts, alpha,
                                          beta))  # Comprovem el minim entre els succesors que tenen valors maxims

                # Comprovem que el score sigui menor a alpha, en cas contrari farem el minim entre beta y score per la seguent iteracio
                if v < alpha:
                    return v
                beta = min(beta, v)

            else:  # En cas que siguin altres ghosts
                v = min(v, self.min_value(succesors, depth, agents + 1, ghosts, alpha,
                                          beta))  # Comprovem el minim entre els succesors que tenen valors minims
                # Comprovem que el score sigui menor a alpha, en cas contrari farem el minim entre beta y score per la seguent iteracio
                if v < alpha:
                    return v
                beta = min(beta, v)

        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = []
        score = -999999999
        alpha = -999999999
        beta = 999999999
        ghosts = gameState.getNumAgents() - 1

        for legalActions in gameState.getLegalActions():  # Iterem per les accions del agent
            succesors = gameState.generateSuccessor(0, legalActions)  # Generem els succesors
            tmp = score
            score = max(score, self.min_value(succesors, self.depth, 1, ghosts, alpha,
                                              beta))  # Comprovem el maxim entre els succesors que tenen valors minims
            if score > tmp:
                actions.append(legalActions)  # Guardem les accions permeses
            if score > beta:
                return score  # En cas que sigui mayor al beta retornem el resultat
            alpha = max(alpha, score)

        while (len(actions)) != 0:  # Retornem totes les accions
            return actions.pop()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Funcio del maxim valor utilitzada unicament pel calcul del pacman
    def expecti_max(self, gameState, depth, ghosts):
        if gameState.isWin() or gameState.isLose() or depth == 0:  # Cas base
            return self.evaluationFunction(gameState)

        score = -999999999  # Score inicial

        for legalActions in gameState.getLegalActions(0):  # Iterem per les accions del pacman
            succesors = gameState.generateSuccessor(0, legalActions)  # Generem els succesors de les accions
            score = max(score, self.exp_value(succesors, depth, 1, ghosts))
        return score

    def exp_value(self, gameState, depth, agents, ghosts):
        if gameState.isWin() or gameState.isLose() or depth == 0:  # Cas base
            return self.evaluationFunction(gameState)

        # Valor esperat i la probabilitat
        expectedValue = 0
        probabilty = 1.0 / len(gameState.getLegalActions(agents))  # Probabilitat de cada accio

        for legalActions in gameState.getLegalActions(agents):  # Iterem per les accions del pacman
            succesors = gameState.generateSuccessor(agents, legalActions)  # Generem els succesors de les accions
            if agents == ghosts:
                curr_expValue = self.expecti_max(succesors, depth - 1, ghosts)
            else:
                curr_expValue = self.exp_value(succesors, depth, agents + 1, ghosts)
            expectedValue += curr_expValue * probabilty
        return expectedValue

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = []
        score = -999999999
        ghosts = gameState.getNumAgents() - 1

        for legalActions in gameState.getLegalActions():  # Iterem per les accions del agent
            succesors = gameState.generateSuccessor(0, legalActions)  # Generem els succesors
            tmp = score
            score = max(score, self.exp_value(succesors, self.depth, 1, ghosts))  # Ens quedem amb el maxim del nimim
            if score > tmp:
                actions.append(legalActions)  # Guardem les accions permeses

        while (len(actions)) != 0:  # Retornem totes les accions
            return actions.pop()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    food = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    currentScore = scoreEvaluationFunction(currentGameState)

    food_list = newFood.asList()
    food_pieces = []
    food_coordinates = []
    currentScore = scoreEvaluationFunction(currentGameState)
    scared_ghost = []
    bad_ghost = []
    minscared = 1000
    minbad = 1000
    closest_food = 999999999

    if currentGameState.isWin():  # Si el següent estat es guanyador retorna el score maxim
        return 999999999
    if currentGameState.isLose():
        return -999999999
    for foods in food_list:  # Iterem per la llista de foods
        food_pieces.append(manhattanDistance(currentPos, foods))  # Afegim les posicions del menjar mes proxim
    min_food = min(food_pieces)  # Agafem el minim dels menjars

    # Agafem la posicio mes propera al menjar
    if min_food < closest_food:
        closest_food = min_food
    food_coordinates.append(closest_food)
    food_pieces = food_coordinates.pop()

    for ghost_states in newGhostStates:  # Per cada estat del fantasmes
        if currentPos == ghost_states.getPosition():  # Comprovem que si colisionen perdin
            return -1
        else:
            if ghost_states.scaredTimer:
                scared_ghost.append(
                    manhattanDistance(currentPos, ghost_states.getPosition()))  # Afegim els scared ghosts
            else:
                bad_ghost.append(manhattanDistance(currentPos, ghost_states.getPosition()))  # Afegim els bad ghosts

    if len(scared_ghost):  # Si encara queden scared ghosts
        minscared = min(scared_ghost)  # Agafem la minima distancia d'aquest
    if len(bad_ghost):  # Si encara queden bad_ghosts
        minbad = min(bad_ghost)  # Agafem la minima distancia d'aquest

    score = currentScore - food_pieces - minscared - minbad
    return score  # Retornem el score


# Abbreviation
better = betterEvaluationFunction
