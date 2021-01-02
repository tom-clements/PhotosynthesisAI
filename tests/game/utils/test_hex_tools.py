from PhotosynthesisAI.game.utils.hex_tools import notation_from_coords


def test_notation_from_coords():
    assert notation_from_coords((3, 0, -3)) == "g4"
    assert notation_from_coords((-3, 0, 3)) == "a1"
    assert notation_from_coords((0, 0, 0)) == "d4"
    assert notation_from_coords((0, 0, 0)) == "d4"
    assert notation_from_coords((-2, 2, 0)) == "b4"
    assert notation_from_coords((2, -1, -1)) == "f3"
    assert notation_from_coords((0, 3, -3)) == "d7"
    assert notation_from_coords((-1, 3, 2)) == "c6"
