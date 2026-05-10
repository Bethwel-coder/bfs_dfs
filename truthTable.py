import pandas as pd

# Create truth table
import itertools

# Generate all combinations
combinations = list(itertools.product([True, False], repeat=3))

# Calculate results
data = []
for P, Q, R in combinations:
    p_and_q = P and Q
    implication = (not p_and_q) or (not R)  # (P∧Q) → ¬R
    data.append({
        'P': P,
        'Q': Q,
        'R': R,
        'P∧Q': p_and_q,
        '¬R': not R,
        '(P∧Q)→¬R': implication
    })

df = pd.DataFrame(data)

# Display truth table
print("\nTruth Table for (P ∧ Q) → ¬R")
print("=" * 60)
print(df.to_string(index=False))

# Classification
print("\n" + "=" * 60)
print("CLASSIFICATION:")
print("=" * 60)

all_true = df['(P∧Q)→¬R'].all()
all_false = ~df['(P∧Q)→¬R'].any()
any_true = df['(P∧Q)→¬R'].any()
any_false = ~df['(P∧Q)→¬R'].all()

print(f"  • Valid (Tautology):     {all_true}")
print(f"  • Unsatisfiable:         {all_false}")
print(f"  • Satisfiable:           {any_true}")
print(f"  • Falsifiable:           {any_false}")

print(f"\n  • Results: {sum(df['(P∧Q)→¬R'])} True / {8 - sum(df['(P∧Q)→¬R'])} False")