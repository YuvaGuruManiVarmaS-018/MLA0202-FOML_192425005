import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create sample CSV training data
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

pd.DataFrame(data).to_csv('training_data.csv', index=False)

# Read CSV file
df = pd.read_csv('training_data.csv')
X = df.iloc[:, :-1].values.tolist()
Y = df.iloc[:, -1].values.tolist()

n = len(X[0])
# S initialized to the first positive training instance if we want, or most specific
S = [None] * n
G = [['?'] * n]

def covers(h, x):
    return all(h[i] == '?' or h[i] == x[i] for i in range(len(h)))

# Initialize S with the first positive training instance
first_pos = True
for x, label in zip(X, Y):
    if label == 'Yes':
        if first_pos:
            S = list(x)
            first_pos = False
        else:
            for i in range(n):
                if S[i] != x[i]:
                    S[i] = '?'
            
        # Remove from G any hypothesis that does not cover x
        G = [g for g in G if covers(g, x)]
        
    else: # Negative instance
        new_G = []
        for g in G:
            if not covers(g, x):
                new_G.append(g)
            else:
                for i in range(n):
                    if g[i] == '?':
                        if S[i] != '?' and S[i] != x[i]:
                            new_g = list(g)
                            new_g[i] = S[i]
                            if new_g not in new_G:
                                new_G.append(new_g)
        
        # Filter G to keep only the most general hypotheses
        G = []
        for g in new_G:
            # g must be more general than S
            if all(g[i] == '?' or g[i] == S[i] for i in range(n)):
                # and not strictly less general than some other g_prime in new_G
                is_minimal = True
                for g_prime in new_G:
                    if g != g_prime:
                        # check if g_prime is strictly more general than g
                        if all(g_prime[i] == '?' or g_prime[i] == g[i] for i in range(n)) and g_prime != g:
                            is_minimal = False
                            break
                if is_minimal and g not in G:
                    G.append(g)

# Convert to set of tuples for standard output formatting
S_set = {tuple(S)}
G_set = {tuple(g) for g in G}

print("Specific Boundary S:")
for h in S_set:
    print(h)

print("\nGeneral Boundary G:")
for h in G_set:
    print(h)

# Plot final boundaries
features = df.columns[:-1].tolist()
all_h = list(S_set) + list(G_set)
labels = ['S'] * len(S_set) + ['G'] * len(G_set)

if all_h:
    fig, ax = plt.subplots(figsize=(10, 4))
    values = [[str(v) if v is not None else 'Ø' for v in h] for h in all_h]

    codes = {'Ø': 0, '?': 1}
    for col in features:
        for v in df[col].unique():
            codes.setdefault(str(v), len(codes))

    matrix = np.array([[codes[v] for v in row] for row in values])

    ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([
        labels[i] + str(i + 1) + ': ' + str(all_h[i])
        for i in range(len(all_h))
    ])
    ax.set_title('Candidate-Elimination Algorithm')
    ax.set_xlabel('Attributes')
    ax.set_ylabel('Final Hypotheses')
    plt.tight_layout()
    plt.show()
