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

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util
random.seed(42)  # For reproducibility
from game import Agent
from pacman import GameState

global_depth = 6

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"

        # Si es un estado de victoria o derrota, devuélvelo directamente
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        # Penalizar si estamos cerca de un fantasma que no está asustado
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        dangerGhosts = [dist for i, dist in enumerate(ghostDistances) if newScaredTimes[i] == 0]
        ghostPenalty = -2 / min(dangerGhosts) if dangerGhosts else 0  # más penalización si está muy cerca

        # Premiar estar cerca de la comida
        foodList = newFood.asList()
        if len(foodList) == 0:
            foodReward = 100
        else:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            foodReward = 1 / min(foodDistances)  # más recompensa si está más cerca

        # Bonus por comer comida
        foodEaten = currentGameState.getNumFood() - successorGameState.getNumFood()
        return successorGameState.getScore() + foodReward + ghostPenalty + foodEaten * 10

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

    def minimax(self, state : GameState, agentIndex : int, depth : int) -> float:
        '''
        Función que implementa el algoritmo min-max
        Parameters:
        -----------
        state : GameState
            El estado del juego.
        agentIndex : int
            Indice del agente (0 = Pacman)
        depth : int
            La profundidad en el árbol de búsqueda.
        Returns:
        --------
        float
            El valor del juego.
        '''
        # Profundidad máxima
        Max_depth = global_depth
        
        # Movimientos posibles
        movements = state.getLegalActions(agentIndex)
        if not movements or Max_depth == depth:
            return self.evaluationFunction(state)

        # Asiganar maximo (Pacman) y minimo (fantasmas)
        n_agents = state.getNumAgents()
        agent_type = ['max'] + ['min'] * (n_agents - 1)
        agent = agent_type[agentIndex]

        # Pacman --> se busca maximizar
        if agent == 'max':
            best = float('-inf') # Valor mínimo
            best_move = None 

            # Recorrer TODOS los movimientos posibles
            for move in movements:
                # Estado sucesor (lo que pasa si Pacman hace el movimiento)
                suc = state.generateSuccessor(agentIndex, move)
                # Llamar a la función minimax recursivamente (se modifica el agente y la profundidad)
                value = self.minimax(suc, (agentIndex + 1) % n_agents, depth + (1 if agentIndex == n_agents - 1 else 0))
                
                # Si se encuentra un mejor valor, se actualiza
                if value > best:
                    best = value
                    best_move = move
            
            # Raiz del árbol: ya sabemos que este movimiento tiene la máxima puntuación
            # --> devolver el movimiento concreto (no la puntuación)
            if depth == 0:
                return best_move
            # Si no es la raíz, devolver el valor máximo (hasta llegar a la raíz)
            return best
        
        # Fantasmas --> se busca minimizar
        else:
            best = float('inf') # Valor máximo
            # Recorrer TODOS los movimientos posibles
            for move in movements:
                # Estado sucesor (lo que pasa si el fantasma hace el movimiento)
                suc = state.generateSuccessor(agentIndex, move)
                value = self.minimax(suc, (agentIndex + 1) % n_agents, depth + (1 if agentIndex == n_agents - 1 else 0))
                # Si se encuentra un mejor valor, se actualiza
                # (en este caso, el mejor valor es el mínimo)
                if value < best:
                    best = value
            # Devolver el valor mínimo (hasta llegar a la raíz)
            # --> No interesa el movimiento de los fantasmas, solo su puntuación
            return best


        
        

        




    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Llamar a la función minimax para obtener la mejor acción
        return self.minimax(gameState, 0, 0)
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self, state: GameState, agentIndex: int, depth: int, alpha: float, beta: float) -> float:

        # Profundidad máxima
        Max_depth = global_depth
        
        # Movimientos posibles
        movements = state.getLegalActions(agentIndex)
        if not movements or Max_depth == depth:
            return self.evaluationFunction(state)

        # Asiganar maximo (Pacman) y minimo (fantasmas)
        n_agents = state.getNumAgents()
        agent_type = ['max'] + ['min'] * (n_agents - 1)
        agent = agent_type[agentIndex]

        # Pacman --> se busca maximizar
        if agent == 'max':
            best = float('-inf') # Valor mínimo
            best_move = None

            # Recorrer TODOS los movimientos posibles
            for move in movements:
                # Estado sucesor (lo que pasa si Pacman hace el movimiento)
                suc = state.generateSuccessor(agentIndex, move)
                # Llamar a la función minimax recursivamente (se modifica el agente y la profundidad)
                value = self.alphaBeta(suc, (agentIndex + 1) % n_agents, depth + (1 if agentIndex == n_agents - 1 else 0), alpha, beta)

                # Si se encuentra un mejor valor, se actualiza
                if value > best:
                    best = value
                    best_move = move

                # Poda alfa-beta
                alpha = max(alpha, best)
                if beta < alpha:
                    break

            # Raiz del árbol: ya sabemos que este movimiento tiene la máxima puntuación
            # --> devolver el movimiento concreto (no la puntuación)
            if depth == 0:
                return best_move
            # Si no es la raíz, devolver el valor máximo (hasta llegar a la raíz)
            return best

        # Fantasmas --> se busca minimizar
        else:
            best = float('inf') # Valor máximo
            # Recorrer TODOS los movimientos posibles
            for move in movements:
                # Estado sucesor (lo que pasa si el fantasma hace el movimiento)
                suc = state.generateSuccessor(agentIndex, move)
                value = self.alphaBeta(suc, (agentIndex + 1) % n_agents, depth + (1 if agentIndex == n_agents - 1 else 0), alpha, beta)
                # Si se encuentra un mejor valor, se actualiza
                # (en este caso, el mejor valor es el mínimo)
                if value < best:
                    best = value

                # Poda alfa-beta
                beta = min(beta, best)
                if beta < alpha:
                    break

            # Devolver el valor mínimo (hasta llegar a la raíz)
            # --> No interesa el movimiento de los fantasmas, solo su puntuación
            return best


    def alphabeta(self, state, depth, agentIndex, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
            return self.evaluationFunction(state)

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + 1

        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)

        if agentIndex == 0:  # Pacman (MAX)
            bestScore = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score = self.alphabeta(successor, nextDepth, nextAgent, alpha, beta)
                if depth == 0:
                    if bestAction is None or score > bestScore:
                        bestAction = action
                bestScore = max(bestScore, score)
                alpha = max(alpha, bestScore)
                if bestScore > beta:
                    break  # poda
            return bestAction if depth == 0 else bestScore
        else:  # Fantasmas (MIN)
            bestScore = float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score = self.alphabeta(successor, nextDepth, nextAgent, alpha, beta)
                bestScore = min(bestScore, score)
                beta = min(beta, bestScore)
                if bestScore < alpha:
                    break  # poda
            return bestScore

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState, 0, 0, float('-inf'), float('inf'))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
            return self.evaluationFunction(state)

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + 1

        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)

        if agentIndex == 0:  # Pacman (MAX)
            bestScore = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score = self.expectimax(successor, nextDepth, nextAgent)
                if depth == 0:
                    if bestAction is None or score > bestScore:
                        bestAction = action
                bestScore = max(bestScore, score)
            return bestAction if depth == 0 else bestScore
        else:  # Fantasmas (valores esperados)
            total = 0
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score = self.expectimax(successor, nextDepth, nextAgent)
                total += score
            return total / len(legalActions)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Datos básicos
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    capsules = currentGameState.getCapsules()

    # Puntaje base del juego actual
    score = currentGameState.getScore()

    # Penalizar comida restante
    score -= 4 * len(foodList)

    # Penalizar cápsulas restantes
    score -= 10 * len(capsules)

    # Comida: premiar estar cerca
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, f) for f in foodList)
        score += 5 / (minFoodDist + 1)

    # Cápsulas: premiar estar cerca
    if capsules:
        minCapsuleDist = min(manhattanDistance(pacmanPos, c) for c in capsules)
        score += 3 / (minCapsuleDist + 1)

    # Fantasmas
    for i, ghost in enumerate(ghostStates):
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
        if scaredTimes[i] > 0:
            # Si está asustado, queremos acercarnos
            score += 2 / (ghostDist + 1)
        else:
            # Si no, penalizar cercanía
            if ghostDist < 2:
                score -= 10 / (ghostDist + 1)

    return score

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()
        
        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
        
        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    score -= 200  # Gran penalización por estar demasiado cerca
        
        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100
        
        return score + neural_score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)