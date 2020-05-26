from isolation import Isolation
from isolation import DebugState
from sample_players import DataPlayer
import random

class CustomPlayer_9(DataPlayer):
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
            0:113,	1:112,	2:111,	3:110,	4:109,	5:108,	6:107,	7:106,	8:105,	9:104,	10:103,
            13:100,	14:99,	15:98,	16:97,	17:96,	18:95,	19:94,	20:93,	21:92,	22:91,	23:90,
            26:87,	27:86,	28:85,	29:84,	30:83,	31:82,	32:81,	33:80,	34:79,	35:78,	36:77,
            39:74,	40:73,	41:72,	42:71,	43:70,	44:69,	45:68,	46:67,	47:66,	48:65,	49:64,
            52:22,	53:21,	54:20,	55:19,	56:18,	57:17,	58:16,	59:15,	60:14,	61:13,	62:12,
            65:48,	66:47,	67:46,	68:45,	69:44,	70:43,	71:42,	72:41,	73:40,	74:39,	75:38,
            78:35,	79:34,	80:33,	81:32,	82:31,	83:30,	84:29,	85:28,	86:27,	87:26,	88:25,
            91:22,	92:21,	93:20,	94:19,	95:18,	96:17,	97:16,	98:15,	99:14,	100:13,	101:12,
            104:9,	105:8,	106:7,	107:6,	108:5,	109:4,	110:3,	111:2,	112:1,	113:0,	114:1
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