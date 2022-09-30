import random
from stat import filemode
import numpy as np
from PongAI import PongAI
import csv

# Hiperparámetros
LR = 0.4
NUM_EPISODES = 5000
DISCOUNT_RATE = 0.1
MAX_EXPLORATION_RATE = 0.9
MIN_EXPLORATION_RATE = 0.1
EXPLORATION_DECAY_RATE = 0.01

# Si deseas o no tener elementos gráficos del juego (más lento si se muestran)
VISUALIZATION = False


def update_exp_rate(n_games):
    return MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE * n_games)


class Agent:
    # Esta clase posee al agente y define sus comportamientos.

    def __init__(self):
        # Creamos la q_table y la inicializamos en 0.
        # IMPLEMENTAR
        self.q_table = None
        if self.q_table is None:
            self.q_table = dict()
            for c1 in range(-5, 6):
                for c2 in range(0, 6):
                    for c3 in range(0, 4):
                        for c4 in range(0, 4):
                            for c5 in range(0, 3):
                                self.q_table[c1, c2, c3, c4, c5] = [np.random.uniform(-5, 0) for i in range(3)]


        # Inicializamos los juegos realizados por el agente en 0.
        self.n_games = 0

        # Inicializamos el exploration rate.
        self.EXPLORATION_RATE = MAX_EXPLORATION_RATE

    def get_state(self, game):
        # Este método consulta al juego por el estado del agente y lo retorna como una tupla.
        state = []
        # Obtenemos la velocidad en y de la pelota
        velocity = int(round(game.ball.y_vel, 0))
        state.append(velocity)

        # Obtenemos el cuadrante de la pelota
        proximity = 5 - int(round(game.ball.x / game.MAX_X) * 5)
        state.append(proximity)

        # Revisamos la posición de la pelota respecto al extremo superior del agente
        if game.ball.y < (game.right_paddle.y):
            if game.right_paddle.y - game.ball.y > game.right_paddle.height:
                up_state = 0
            else:
                up_state = 1
        else:
            if game.ball.y - game.right_paddle.y < game.right_paddle.height:
                up_state = 2
            else:
                up_state = 3
        state.append(up_state)

        # Revisamos la posición de la pelota respecto al extremo inferior del agente 
        if game.ball.y < (game.right_paddle.y + game.right_paddle.height):
            if game.right_paddle.y + game.right_paddle.height - game.ball.y > game.right_paddle.height:
                down_state = 0
            else:
                down_state = 1
        else:
            if game.ball.y - game.right_paddle.y - game.right_paddle.height < game.right_paddle.height:
                down_state = 2
            else:
                down_state = 3
        state.append(down_state)

        # Número de botes contra la pared que ha dado la pelota
        bounces = game.ball.bounces
        state.append(bounces)

        return tuple(state)

    def get_action(self, state):
        # Este método recibe una estado del agente y retorna un entero con el índice de la acción correspondiente.

        # IMPLEMENTAR LISTO
        if np.random.random() > self.EXPLORATION_RATE:
            action = np.argmax(self.q_table[state])
        else:
            # action = np.random.randint(0, 4)
            action = np.random.randint(0, 3)
        return action


def train():
    # Esta función es la encargada de entrenar al agente.

    # Las siguientes variables nos permitirán llevar registro del desempeño del agente.
    plot_scores = []
    plot_mean_scores = []
    mean_score = 0
    total_score = 0
    record = 0
    period_steps = 0
    period_score = 0

    # Instanciamos al agente o lo cargamos desde un pickle.
    agent = Agent()

    # Instanciamos el juego. El bool 'vis' define si queremos visualizar el juego o no.
    # Visualizarlo lo hace mucho más lento.
    game = PongAI(vis=VISUALIZATION)

    # Inicializamos los pasos del agente en 0.
    steps = 0

    while True:
        # Obtenemos el estado actual.
        state = agent.get_state(game)
        # Generamos la acción correspondiente al estado actual.
        move = agent.get_action(state)
        # Ejecutamos la acción.
        reward, done, score = game.play_step(move)

        # Obtenemos el nuevo estado.
        # IMPLEMENTAR LISTO
        new_state = agent.get_state(game)

        # Actualizamos la q-table.
        # IMPLEMENTAR
        max_future_q = np.max(agent.q_table[new_state])
        current_q = agent.q_table[state][move]
        new_q = (1 - LR) * current_q + LR * (reward + DISCOUNT_RATE * max_future_q)
        agent.q_table[state][move] = new_q

        # En caso de terminar el juego.
        if done:
            # Actualizamos el exploration rate.
            # IMPLEMENTAR LISTO
            agent.EXPLORATION_RATE = update_exp_rate(agent.n_games)

            # Reiniciamos el juego.
            game.reset()

            # Actualizamos los juegos jugados por el agente.
            agent.n_games += 1

            # Imprimimos el desempeño del agente cada 100 juegos.
            if agent.n_games % 100 == 0:
                # La siguiente línea guarda la QTable en un archivo (para poder ser accedida posteriormente)
                np.save("q_table.npy", agent.q_table)

                # Información relevante sobre los últimos 100 juegos
                print('Game', agent.n_games, 'Mean Score', period_score/100, 'Record:', record, "EXP_RATE:", agent.EXPLORATION_RATE, "STEPS:", period_steps/100)
                record = 0
                period_score = 0
                period_steps = 0

            # Actualizamos el record del agente.
            if score > record:
                record = score

            # Actualizamos nuestros indicadores.
            period_steps += steps
            period_score += score
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            steps = 0

            # En caso de alcanzar el máximo de juegos cerramos el loop
            if agent.n_games == NUM_EPISODES:
                break

        else:
            steps += 1

    f = open('q_table.csv', 'w')

    with f:
        writer = csv.writer(f)

        for row in agent.q_table:
            writer.writerow(row + (tuple(agent.q_table[row])))


if __name__ == '__main__':
    train()
