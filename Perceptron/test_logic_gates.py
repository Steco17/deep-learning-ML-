import numpy as np

# Import functions from perceptron.py
from perceptron import step, perceptron


# -------------------------------------------------
# Gate configurations (weights + bias)
# -------------------------------------------------
W_AND = np.array([1, 1])
B_AND = -1.5

W_OR = np.array([1, 1])
B_OR = -0.5

W_NAND = np.array([-1, -1])
B_NAND = 1.5

W_NOR = np.array([-1, -1])
B_NOR = 0.5

W_NOT = np.array([-1])
B_NOT = 0.5


# -------------------------------------------------
# AND gate
# -------------------------------------------------
def test_and_gate():
    tests = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ]

    for x, expected in tests:
        result = perceptron(np.array(x), W_AND, B_AND)
        assert result == expected, f"AND failed for {x}"


# -------------------------------------------------
# OR gate
# -------------------------------------------------
def test_or_gate():
    tests = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
    ]

    for x, expected in tests:
        result = perceptron(np.array(x), W_OR, B_OR)
        assert result == expected, f"OR failed for {x}"


# -------------------------------------------------
# NAND gate
# -------------------------------------------------
def test_nand_gate():
    tests = [
        ([0, 0], 1),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    for x, expected in tests:
        result = perceptron(np.array(x), W_NAND, B_NAND)
        assert result == expected, f"NAND failed for {x}"


# -------------------------------------------------
# NOR gate
# -------------------------------------------------
def test_nor_gate():
    tests = [
        ([0, 0], 1),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 0),
    ]

    for x, expected in tests:
        result = perceptron(np.array(x), W_NOR, B_NOR)
        assert result == expected, f"NOR failed for {x}"


# -------------------------------------------------
# NOT gate
# -------------------------------------------------
def test_not_gate():
    tests = [
        (0, 1),
        (1, 0),
    ]

    for x, expected in tests:
        result = perceptron(np.array([x]), W_NOT, B_NOT)
        assert result == expected, f"NOT failed for {x}"


# -------------------------------------------------
# XOR gate (2-layer perceptron)
# -------------------------------------------------
def xor(x):
    # First layer
    or_out = perceptron(x, W_OR, B_OR)
    nand_out = perceptron(x, W_NAND, B_NAND)

    # Second layer
    return perceptron(
        np.array([or_out, nand_out]),
        W_AND,
        B_AND
    )


def test_xor_gate():
    tests = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    for x, expected in tests:
        result = xor(np.array(x))
        assert result == expected, f"XOR failed for {x}"


# -------------------------------------------------
# Manual runner (optional)
# -------------------------------------------------
if __name__ == "__main__":
    test_and_gate()
    test_or_gate()
    test_nand_gate()
    test_nor_gate()
    test_not_gate()
    test_xor_gate()
    print("All perceptron gate tests passed!")
    