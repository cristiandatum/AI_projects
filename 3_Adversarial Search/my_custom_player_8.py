from isolation import Isolation
from isolation import DebugState
from sample_players import DataPlayer
import random

class CustomPlayer_8(DataPlayer):
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
            0:114,	1:113,	2:112,	3:111,	4:110,	5:109,	6:108,	7:107,	8:106,	9:105,	10:104,
            13:101,	14:100,	15:99,	16:98,	17:97,	18:96,	19:95,	20:94,	21:93,	22:92,	23:91,
            26:88,	27:87,	28:86,	29:85,	30:84,	31:83,	32:82,	33:81,	34:80,	35:79,	36:78,
            39:75,	40:74,	41:73,	42:72,	43:71,	44:70,	45:69,	46:68,	47:67,	48:66,	49:65,
            52:23,	53:22,	54:21,	55:20,	56:19,	57:18,	58:17,	59:16,	60:15,	61:14,	62:13,
            65:49,	66:48,	67:47,	68:46,	69:45,	70:44,	71:43,	72:42,	73:41,	74:40,	75:39,
            78:36,	79:35,	80:34,	81:33,	82:32,	83:31,	84:30,	85:29,	86:28,	87:27,	88:26,
            91:23,	92:22,	93:21,	94:20,	95:19,	96:18,	97:17,	98:16,	99:15,	100:14,	101:13,
            104:10,	105:9,	106:8,	107:7,	108:6,	109:5,	110:4,	111:3,	112:2,	113:1,	114:0
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