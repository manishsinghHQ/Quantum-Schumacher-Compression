import streamlit as st
import heapq
import numpy as np
from typing import Dict, Optional

st.set_page_config(page_title="Quantum Schumacher Compression", layout="wide")

st.title("Schumacher Compression — Quantum Huffman Encoding")

st.markdown("""
This app demonstrates **Quantum Source Compression (Schumacher Compression)**.

Pipeline:

Classical Probability Distribution  
↓  
Huffman Coding  
↓  
Quantum Amplitude Encoding  
↓  
Von Neumann Entropy (Compression Limit)
""")

# ---------------------------------------------------
# Huffman Tree
# ---------------------------------------------------

class HuffmanNode:
    def __init__(self, symbol=None, prob: float = 0.0,
                 left=None, right=None):
        self.symbol = symbol
        self.prob = prob
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(probs: Dict[str, float]):
    heap = [HuffmanNode(symbol=s, prob=p) for s, p in probs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(prob=left.prob + right.prob,
                                         left=left, right=right))
    return heap[0]


def huffman_codes(root, prefix="", codes=None):
    if codes is None:
        codes = {}

    if root.symbol is not None:
        codes[root.symbol] = prefix or "0"
    else:
        huffman_codes(root.left, prefix + "0", codes)
        huffman_codes(root.right, prefix + "1", codes)

    return codes


# ---------------------------------------------------
# Entropy
# ---------------------------------------------------

def shannon_entropy(probs):
    return -sum(p * np.log2(p) for p in probs.values() if p > 0)


def von_neumann_entropy(density_matrix):
    eigvals = np.linalg.eigvalsh(density_matrix)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log2(eigvals))


def huffman_avg_length(probs, codes):
    return sum(probs[s] * len(codes[s]) for s in codes)


# ---------------------------------------------------
# Quantum amplitudes
# ---------------------------------------------------

def huffman_amplitudes(probs, codes):

    n = max(len(c) for c in codes.values())
    dim = 2 ** n

    vec = np.zeros(dim)

    for symbol, code in codes.items():
        idx = int(code.ljust(n, "0"), 2)
        vec[idx] += np.sqrt(probs[symbol])

    vec = vec / np.linalg.norm(vec)

    return vec, n


# ---------------------------------------------------
# UI: User Input
# ---------------------------------------------------

st.sidebar.header("Probability Input")

symbols = st.sidebar.slider("Number of Symbols", 2, 6, 4)

probs = {}
for i in range(symbols):
    p = st.sidebar.number_input(f"Probability P(Symbol {i})",
                                min_value=0.0,
                                max_value=1.0,
                                value=round(1/symbols, 2))
    probs[f"S{i}"] = p

# Normalize probabilities
total = sum(probs.values())

if total == 0:
    st.warning("Probabilities cannot all be zero.")
    st.stop()

probs = {k: v/total for k, v in probs.items()}

st.subheader("Normalized Probabilities")

st.write(probs)

# ---------------------------------------------------
# Run compression
# ---------------------------------------------------

tree = build_huffman_tree(probs)
codes = huffman_codes(tree)

amps, n_qubits = huffman_amplitudes(probs, codes)

rho = np.diag(list(probs.values()))

shannon = shannon_entropy(probs)
vne = von_neumann_entropy(rho)
avg_len = huffman_avg_length(probs, codes)

compression_ratio = shannon / avg_len

# ---------------------------------------------------
# Results
# ---------------------------------------------------

st.header("Huffman Codes")

table_data = []

for s in probs:
    table_data.append({
        "Symbol": s,
        "Probability": probs[s],
        "Code": codes[s],
        "Amplitude √p": np.sqrt(probs[s])
    })

st.table(table_data)

# ---------------------------------------------------
# Entropy Metrics
# ---------------------------------------------------

st.header("Information Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Shannon Entropy H(X)", f"{shannon:.4f} bits")

col2.metric("Von Neumann Entropy S(ρ)", f"{vne:.4f} qubits")

col3.metric("Avg Huffman Length", f"{avg_len:.4f} bits")

st.metric("Compression Ratio", f"{compression_ratio:.4f}")

st.success(f"Schumacher Limit ≥ {vne:.4f} qubits per symbol")

# ---------------------------------------------------
# Quantum State
# ---------------------------------------------------

st.header("Quantum State |ψ⟩")

state_data = []

for i, amp in enumerate(amps):
    if abs(amp) > 1e-6:
        state_data.append({
            "Basis State": format(i, f"0{n_qubits}b"),
            "Amplitude": amp
        })

st.table(state_data)

# ---------------------------------------------------
# Optional Qiskit circuit
# ---------------------------------------------------

try:
    from qiskit import QuantumCircuit

    st.header("Qiskit Circuit")

    qc = QuantumCircuit(n_qubits)
    qc.initialize(amps.tolist(), range(n_qubits))

    st.text(qc)

except:
    st.info("Install Qiskit to view the circuit")
