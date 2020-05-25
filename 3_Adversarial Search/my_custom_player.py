from isolation import Isolation
from isolation import DebugState
from sample_players import DataPlayer
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal alpha_beta minimax search move at a search 
        # depth of 4 plies using minimax with alpha-beta pruning.
        
        #my_custom_player moves first:
        if state.ply_count ==0:
            self.queue.put(1)

        #my_custom_player moves second:
        elif state.ply_count ==1:
            all_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 
            19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 
            59, 60, 61, 62, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 78, 79, 
            80, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 
            99, 100, 101, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]

            #map to use against opponent's first move:
            move_map={
            0:14,	1:15,	2:16,	3:15,	4:18,	5:17,	6:18,	7:19,	8:20,	9:21,	10:22,
            13:27,	14:28,	15:29,	16:28,	17:31,	18:30,	19:31,	20:32,	21:33,	22:34,	23:35,
            26:14,	27:41,	28:42,	29:43,	30:44,	31:45,	32:46,	33:45,	34:46,	35:47,	36:22,
            39:53,	40:54,	41:55,	42:56,	43:57,	44:56,	45:57,	46:58,	47:59,	48:60,	49:61,
            52:40,	53:67,	54:42,	55:69,	56:70,	57:71,	58:70,	59:71,	60:46,	61:47,	62:48,
            65:53,	66:54,	67:55,	68:56,	69:57,	70:84,	71:85,	72:58,	73:59,	74:60,	75:61,
            78:66,	79:67,	80:68,	81:69,	82:70,	83:69,	84:70,	85:71,	86:72,	87:73,	88:74,
            91:79,	92:80,	93:81,	94:82,	95:83,	96:82,	97:83,	98:84,	99:85,	100:86,	101:87,
            104:92,	105:93,	106:94,	107:95,	108:96,	109:97,	110:98,	111:97,	112:98,	113:99,	114:100
            }

            #gives the used square
            opp_move_1= list(set(all_actions)^set(state.actions()))
            my_move_1=move_map[opp_move_1[0]]
            
            self.queue.put(my_move_1)

        #progressively go deeper as the number of state options is reduced.    
        elif state.ply_count >65:
            self.queue.put(self.alpha_beta_search(state, depth=5))

        elif state.ply_count >50 and state.ply_count<=65:
            self.queue.put(self.alpha_beta_search(state, depth=4))

        else:
            self.queue.put(self.alpha_beta_search(state, depth=3))

    def alpha_beta_search(self, state, depth):

        def _min_value(state, alpha,beta,depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)

            v=float("inf")
            for a in state.actions():
                v=min(v,_max_value(state.result(a),alpha,beta, depth-1))
                if v<=alpha:
                    return v
                beta=min(beta,v)
            return v

        def _max_value(state, alpha,beta,depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            v = float("-inf")
            for a in state.actions():
                v = max(v, _min_value(state.result(a), alpha,beta,depth - 1))
                if v>=beta:
                    return v
                beta=min(beta,v)
            return v

        alpha=float("-inf")
        beta=float("inf")
        best_score=float("-inf")
        best_move=None 

        for a in state.actions():
            v=_min_value(state.result(a),alpha,beta,depth)
            alpha=max(alpha,v)
            if v>=best_score:
                best_score=v
                best_move=a
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)