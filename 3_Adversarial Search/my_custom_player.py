
import pickle
from isolation import Isolation
from sample_players import DataPlayer

for play in range(10):
    state = Isolation()
    my_data = {state: 57}  # opening book always chooses the middle square on an open board
    pickle_out = open('data.pickle', 'wb')
    pickle.dump(my_data,pickle_out)
    
pickle_out.close()


pickle_in=open('data.pickle','rb')
example=pickle.load(pickle_in)    

print(example)
print(DataPlayer)

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
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        self.queue.put(random.choice(state.actions()))

        def _minimax(self, state, depth):

            def _min_value(state, depth):
                if state.terminal_test(): return state.utility(self.player_id)
                if depth <= 0: return self.score(state)
                value = float("inf")
                for action in state.actions():
                    value = min(value, _max_value(state.result(action), depth - 1))
                return value

            def _max_value(state, depth):
                if state.terminal_test(): return state.utility(self.player_id)
                if depth <= 0: return self.score(state)
                value = float("-inf")
                for action in state.actions():
                    value = max(value, _min_value(state.result(action), depth - 1))
                return value

            return max(state.actions(), key=lambda x: _min_value(state.result(x), depth - 1))
