import numpy as np


class TicTacToe:
    def __init__(self, start_player: str = 'O', alpha_beta_pruning=True, game_state: np.array = None):
        if game_state is None:
            self.game_state = np.array([['.']*3 for _ in range(3)])
        else:
            self.game_state = game_state
        self.symbols = {'X': 1, 'O': 0, '.': float('inf')}
        self.player = start_player
        self.alpha_beta_pruning = alpha_beta_pruning
        self.computations_counter = None
        self.minimax()

    @staticmethod
    def count_computations():
        """
        A Counter to count number of computations
        :return:
        """
        count = 0
        while True:
            count += 1
            yield count

    def get_available_moves(self) -> list:
        """
        This method will calculated the available moves in the game board,
        and return them in a list of tuples
        :return:
        """
        available_moves = np.where(self.game_state == '.')
        available_moves = [(available_moves[0][i], available_moves[1][i]) for i in range(available_moves[0].size)]
        return available_moves

    def get_game_result(self) -> str:
        """
        This method will calculate the game's result by checking the current game state
        :return: str ('O' : Computer wins; 'X' : Human Wins; '.' : Game Draw; '?' : Undecidable)
        """
        numerical_game_map = np.array(list(map(lambda i: [self.symbols[j] for j in i], self.game_state)))

        horizontal_sum = numerical_game_map.sum(axis=1)  # Horizontal
        vertical_sum = numerical_game_map.sum(axis=0)  # Vertical
        diagonal_1_sum = numerical_game_map.diagonal().sum()  # Diagonal 1
        diagonal_2_sum = np.flipud(numerical_game_map).diagonal().sum()  # Diagonal 2

        if np.where(horizontal_sum == 0)[0].size > 0 \
                or np.where(vertical_sum == 0)[0].size > 0 \
                or diagonal_1_sum == 0 \
                or diagonal_2_sum == 0:
            return 'O'
        elif np.where(horizontal_sum == 3)[0].size > 0 \
                or np.where(vertical_sum == 3)[0].size > 0 \
                or diagonal_1_sum == 3 \
                or diagonal_2_sum == 3:
            return 'X'
        elif len(self.get_available_moves()) == 0:
            return '.'
        else:
            return '?'

    def __max(self) -> tuple:
        """
        Maximizing Function
        :return: a tuple containing cost of the best move and the move's position (value, (row, column))
        """
        available_moves = self.get_available_moves()
        s = self.get_game_result()
        if s == 'X':
            return -1 * (len(available_moves)+1), (None, None)
        elif s == 'O':
            return 1 * (len(available_moves)+1), (None, None)
        elif s == '.':
            return 0, (None, None)

        max_value = (float('-inf'), (None, None))
        for move in available_moves:
            next(self.computations_counter)
            self.game_state[move[0], move[1]] = 'O'
            value = self.__min()
            max_value = max(max_value, (value[0], (move[0], move[1])))
            self.game_state[move[0], move[1]] = '.'  # Undo move
        return max_value

    def __min(self) -> tuple:
        """
        Minimizing Function
        :return: a tuple containing cost of the best move and the move's position (value, (row, column))
        """
        available_moves = self.get_available_moves()
        s = self.get_game_result()
        if s == 'X':
            return -1 * (len(available_moves)+1), (None, None)
        elif s == 'O':
            return 1 * (len(available_moves)+1), (None, None)
        elif s == '.':
            return 0, (None, None)

        min_value = (float('inf'), (None, None))
        for move in available_moves:
            next(self.computations_counter)
            self.game_state[move[0], move[1]] = 'X'
            value = self.__max()
            min_value = min(min_value, (value[0], (move[0], move[1])))
            self.game_state[move[0], move[1]] = '.'  # Undo move
        return min_value

    def __max_alpha_beta(self, alpha=float('-inf'), beta=float('inf')) -> tuple:
        """
        Maximizing Function with alpha beta pruning
        :param alpha: maximum available value of Maximizing player. Default: -infinity
        :param beta: minimum available value of Minimizing player. Default: +infinity
        :return: a tuple containing cost of the best move and the move's position (value, (row, column))
        """
        available_moves = self.get_available_moves()
        s = self.get_game_result()
        if s == 'X':
            return -1 * (len(available_moves) + 1), (None, None)
        elif s == 'O':
            return 1 * (len(available_moves) + 1), (None, None)
        elif s == '.':
            return 0, (None, None)

        max_value = (float('-inf'), (None, None))
        for move in available_moves:
            next(self.computations_counter)
            self.game_state[move[0], move[1]] = 'O'
            value = self.__min_alpha_beta(alpha, beta)
            max_value = max(max_value, (value[0], (move[0], move[1])))
            self.game_state[move[0], move[1]] = '.'  # Undo move
            alpha = max(max_value[0], alpha)
            if beta < alpha:
                break

        return max_value

    def __min_alpha_beta(self, alpha=float('-inf'), beta=float('inf')) -> tuple:
        """
        Minimizing Function with alpha beta pruning
        :param alpha: maximum available value of Maximizing player. Default: -infinity
        :param beta: minimum available value of Minimizing player. Default: +infinity
        :return: a tuple containing cost of the best move and the move's position (value, (row, column))
        """
        available_moves = self.get_available_moves()
        s = self.get_game_result()
        if s == 'X':
            return -1 * (len(available_moves) + 1), (None, None)
        elif s == 'O':
            return 1 * (len(available_moves) + 1), (None, None)
        elif s == '.':
            return 0, (None, None)

        min_value = (float('inf'), (None, None))
        for move in available_moves:
            next(self.computations_counter)
            self.game_state[move[0], move[1]] = 'X'
            value = self.__max_alpha_beta(alpha, beta)
            min_value = min(min_value, (value[0], (move[0], move[1])))
            self.game_state[move[0], move[1]] = '.'  # Undo move
            beta = min(beta, min_value[0])
            if beta < alpha:
                break

        return min_value

    def minimax(self):
        """
        Minimax function - the driver function for the game
        :return: None
        """
        while True:
            print(self.game_state)
            result = self.get_game_result()
            if result == 'X':
                print("Human Wins!")
                return
            elif result == 'O':
                print("Computer Wins!")
                return
            elif result == '.':
                print("Game Draw")
                return

            if self.player == 'O':
                self.computations_counter = self.count_computations()
                (_, (row, column)) = self.__max_alpha_beta() if self.alpha_beta_pruning else self.__max()
                print("Computer's Move: ({}, {}), after doing {} computations"
                      .format(row, column, next(self.computations_counter)))
                self.game_state[row, column] = 'O'
                self.player = 'X'
            elif self.player == 'X':
                row, column = tuple(map(lambda x: int(x), input("Your move (row, column):")
                                        .replace(' ', '').split(',')))
                self.game_state[row, column] = 'X'
                self.player = 'O'
