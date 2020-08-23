class Tree:
    max_size = 3

    def __init__(self, owner, size, bought, tree_type, score, tile=None, cost=None):
        self.owner = owner
        self.size = size
        self.shadow = size
        self.tile = tile
        self.bought = bought
        self.cost = cost
        self.tree_type = tree_type
        self.score = score

    def can_grow(self):
        return self.size < self.max_size