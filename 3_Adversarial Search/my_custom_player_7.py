from isolation import Isolation
from isolation import DebugState
from sample_players import DataPlayer
import random

class CustomPlayer_7(DataPlayer):
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
        # return the optimal alpha_beta minimax search move at a variable 
        # depth plies using minimax with alpha-beta pruning and first best move.
        
        if state.ply_count ==0:
            self.queue.put(random.choice(state.actions()))

        elif state.ply_count ==1:

            all_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 
            19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 
            59, 60, 61, 62, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 78, 79, 
            80, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 
            99, 100, 101, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]

            #map to use against opponent's first move:
            move_map={
            0:13,	1:0,	2:1,	3:2,	4:3,	5:4,	6:5,	7:6,	8:7,	9:10,	10:23,
            13:26,	14:27,	15:28,	16:29,	17:30,	18:31,	19:32,	20:33,	21:34,	22:35,	23:36,
            26:39,	27:40,	28:41,	29:42,	30:43,	31:44,	32:45,	33:46,	34:47,	35:48,	36:49,
            39:52,	40:53,	41:54,	42:55,	43:56,	44:57,	45:58,	46:59,	47:60,	48:61,	49:62,
            52:65,	53:66,	54:67,	55:68,	56:69,	57:70,	58:71,	59:72,	60:73,	61:74,	62:75,
            65:52,	66:53,	67:54,	68:55,	69:56,	70:57,	71:58,	72:59,	73:60,	74:61,	75:62,
            78:65,	79:66,	80:67,	81:68,	82:69,	83:70,	84:71,	85:72,	86:73,	87:74,	88:75,
            91:78,	92:79,	93:80,	94:81,	95:82,	96:83,	97:84,	98:85,	99:86,	100:87,	101:88,
            104:91,	105:104,	106:105,	107:106,	108:107,	109:108,	110:109,	111:110,	112:111,	113:112,	114:101
            }
            #gives the used square
            opp_move_1= list(set(all_actions)^set(state.actions()))
            my_move_1=move_map[opp_move_1[0]]
            
            self.queue.put(my_move_1)

        else:
            depth=3
            depth_limit=10
            search_best_score=float(-10)
            search_best_move=None

            best_score,best_move=self.alpha_beta_search(state,depth) 
            self.queue.put(best_move)

            search_best_score, search_best_move=best_score,best_move

            for d in range (4,depth_limit,1):

                best_score,best_move=self.alpha_beta_search(state,d)
#                search_best_move=best_move
#                print('move number', state.ply_count, 'best move', search_best_move, 'best score', best_score)

                if best_score>search_best_score:
                    search_best_score =best_score
                    search_best_move=best_move
#                print(search_best_move, search_best_score)
                self.queue.put(search_best_move)

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
                alpha=max(alpha,v)
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
        return best_score, best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)