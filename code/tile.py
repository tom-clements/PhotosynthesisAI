class Tile:
    def __init__(self, tree, coords, index):
        self.tree = tree
        self.coords = coords
        self.index = index
        self.is_locked = False