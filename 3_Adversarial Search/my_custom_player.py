from isolation import Isolation
from isolation import DebugState #me added
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
        # return the optimal alpha_beta minimax search move at a fixed search 
        # depth of 4 plies using minimax with alpha-beta pruning.
        
        #print(state.ply_count) #added me

#        debug_board = DebugState.from_state(state) #me added
#        print(debug_board) #me added
#        print(state.actions()) #me added

        #my_custom_player moves first:
        if state.ply_count ==0:
#            print ("this is my ply_count:", state.ply_count)
#            print(state.actions())
#            debug_board=DebugState.from_state(state)
#            print(debug_board)
#            self.queue.put(random.choice(state.actions()))
#            state.actions()
#            select the middle square in 11 x 9 board
            self.queue.put(57)

        #my_custom_player moves second:
        elif state.ply_count ==1:
            all_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 
            19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 
            59, 60, 61, 62, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 78, 79, 
            80, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 
            99, 100, 101, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]

            #gives the used square
            opp_move_1= list(set(all_actions)^set(state.actions()))
            
            #if opponent 1st move is in a
            if opp_move_1 == 0:
                my_move_1 = 14
            elif opp_move_1==10:
                my_move_1 = 22
            elif opp_move_1==104:
                my_move_1=92
            elif opp_move_1==114:
                my_move_1=100

            #find location of 1st opponent move and add +2 so it lies in the 
            #same coloured square
           
            my_move_1= all_actions[all_actions.index((opp_move_1[0])+2)]
            self.queue.put(my_move_1)

        elif state.ply_count >60:
            self.queue.put(self.alpha_beta_search(state, depth=4))

        else:
            self.queue.put(self.alpha_beta_search(state, depth=4))

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
        best_move=None # unused variable

        for a in state.actions():
            v=_min_value(state.result(a),alpha,beta,depth)
            alpha=max(alpha,v)
            if v>best_score:
                best_score=v
                best_move=a

#        board = Isolation() #me added
#        debug_board = DebugState.from_state(board) #me added
#        print(debug_board.bitboard_string) #me added
        
#        print(best_move)
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)