
import math
import glicko2

def test_glicko_library():
    p = glicko2.Player(1500, 350, 0.06)
    # Internal scale: rating=0, rd=350/173.7178 = 2.0147
    # If we update with one game
    try:
        p.update_player([1500], [350], [1])
        print(f"Update successful: {p.rating:.2f}, {p.rd:.2f}")
    except Exception as e:
        print(f"Update failed: {e}")

    # Now try with a high rating
    p2 = glicko2.Player(2500, 350, 0.06)
    # Internal scale: rating = (2500-1500)/173.7178 = 5.756
    try:
        p2.update_player([2500], [350], [1])
        print(f"Update successful (high rating): {p2.rating:.2f}, {p2.rd:.2f}")
    except Exception as e:
        print(f"Update failed (high rating): {e}")

if __name__ == "__main__":
    test_glicko_library()
