import numpy as np

# -------------------------------------------------
# Step activation function
# -------------------------------------------------
# Converts a real number into a binary output
# If z >= 0 → 1
# If z < 0  → 0
def step(z):
    return 1 if z >= 0 else 0


# -------------------------------------------------
# Perceptron function
# -------------------------------------------------
# x : input vector (e.g. [0, 1])
# w : weight vector
# b : bias
def perceptron(x, w, b):
    z = np.dot(x, w) + b
    return step(z)


# -------------------------------------------------
# Test helper function
# -------------------------------------------------
def test_gate(gate_name, w, b):
    print(f"\n{gate_name} gate")
    tests = [
        ([0, 0], None),
        ([0, 1], None),
        ([1, 0], None),
        ([1, 1], None),
    ]

    for x, _ in tests:
        pred = perceptron(np.array(x), w, b)
        print(f"{x} -> {pred}")


# =================================================
# LOGIC GATES
# =================================================

# -----------------
# AND gate
# Output = 1 only if both inputs are 1
# -----------------
# z = x1 + x2 - 1.5
# Only [1,1] gives z >= 0
w_and = np.array([1, 1])
b_and = -1.5
test_gate("AND", w_and, b_and)


# -----------------
# OR gate
# Output = 1 if at least one input is 1
# -----------------
# z = x1 + x2 - 0.5
# Any input with a 1 gives z >= 0
w_or = np.array([1, 1])
b_or = -0.5
test_gate("OR", w_or, b_or)


# -----------------
# NAND gate
# Output = 0 only if both inputs are 1
# -----------------
# Inverse of AND
# z = -x1 - x2 + 1.5
w_nand = np.array([-1, -1])
b_nand = 1.5
test_gate("NAND", w_nand, b_nand)


# -----------------
# NOR gate
# Output = 1 only if both inputs are 0
# -----------------
# Inverse of OR
# z = -x1 - x2 + 0.5
w_nor = np.array([-1, -1])
b_nor = 0.5
test_gate("NOR", w_nor, b_nor)


# -----------------
# NOT gate (single input)
# -----------------
# NOT x = 1 when x = 0, else 0
# z = -x + 0.5
print("\nNOT gate")
w_not = np.array([-1])
b_not = 0.5

for x in [0, 1]:
    pred = perceptron(np.array([x]), w_not, b_not)
    print(f"{x} -> {pred}")


# =================================================
# XOR gate (cannot be done with ONE perceptron)
# =================================================
print("\nXOR gate (using 2-layer perceptron)")

# XOR = (x OR y) AND (x NAND y)

# First layer
def xor(x):
    x = np.array(x)

    # OR
    or_out = perceptron(x, w_or, b_or)

    # NAND
    nand_out = perceptron(x, w_nand, b_nand)

    # Second layer: AND on outputs of first layer
    return perceptron(
        np.array([or_out, nand_out]),
        w_and,
        b_and
    )

# Test XOR
xor_tests = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for x, expected in xor_tests:
    print(f"{x} -> {xor(x)} (expected {expected})")
