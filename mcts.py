import math

# W: total weight
# N: number visit
#

class Node:
    def __init__(self, state, parent, prior=0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.is_expanded = False
        self.children = {}
        self.N = 0
        self.W = 0

    @property
    def Q(self):
        if self.N == 0:
            return self.W/1.0
        else:
            return self.W/self.W

    @property
    def U(self):  # returns float
        return (math.sqrt(self.parent.number_visits)
                * self.prior / (1 + self.number_visits))

    def best_child(self):
        return max(self.children.values(), key=lambda n: n.Q()+n.U())

    def select(self):
        node = self
        while node.is_expanded:
            node = node.best_child()
        return node

    def expand(self, child_prior_ps):
        for action, prior_p in enumerate(child_prior_ps):
            self.children[action] = Node(state=self.state.take_move(action), parent=self, prior=prior_p)
        self.is_expanded = True

    def back_update(self, value):
        node = self
        # TODO: 用 game 里面的 who player lai 确定正负
        factor = 1
        # check if node is root
        while node is not None:
            node.N += 1
            node.W += (value*factor)
            node = node.parent
            factor = factor * -1




