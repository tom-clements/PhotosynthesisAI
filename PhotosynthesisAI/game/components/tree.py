class Tree:
    max_size = 3

    def __init__(
        self,
        id: int,
        owner: int,
        size: int,
        is_bought: bool,
        tree_type: str,
        score: int,
        tile: "Tile" = None,
        cost: int = None,
    ):
        self.id = id
        self.owner = owner
        self.size = size
        self.shadow = size
        self.tile = tile
        self.is_bought = is_bought
        self.cost = cost
        self.tree_type = tree_type
        self.score = score
        self.is_deleted = False

    def can_grow(self) -> bool:
        return self.size < self.max_size
