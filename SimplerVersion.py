import math
from collections import defaultdict, Counter


def read_fasta(path):
    sequences = []
    seq = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append("".join(seq).upper())
                    seq = []
            else:
                seq.append(line)

        if seq:
            sequences.append("".join(seq).upper())

    return sequences


def estimate_markov_probs(sequences, m, pseudocount=0.5):
    counts = defaultdict(Counter)

    for seq in sequences:
        for i in range(m, len(seq)):
            ctx = seq[i - m:i] if m > 0 else ""
            base = seq[i]

            if (len(ctx) != m) or (base not in "ACGT") or any(ch not in "ACGT" for ch in ctx):
                continue

            counts[ctx][base] += 1

    probs = {}

    for ctx, counter in counts.items():
        total = sum(counter[b] for b in "ACGT") + 4 * pseudocount
        probs[ctx] = {
            b: (counter[b] + pseudocount) / total
            for b in "ACGT"
        }

    return probs


def score_log_likelihood(seq, m, probs):
    s = 0.0
    for i in range(m, len(seq)):
        ctx = seq[i - m:i] if m > 0 else ""
        base = seq[i]

        if (len(ctx) != m) or (base not in "ACGT") or any(ch not in "ACGT" for ch in ctx):
            continue

        p_ctx = probs.get(ctx)
        p = (1.0 / 4.0) if p_ctx is None else p_ctx[base]

        s += math.log(p)

    return s

#Main code

fasta_file = input("Enter FASTA file path: ")
m = int(input("Markov model order m: "))

sequences = read_fasta(fasta_file)

probs = estimate_markov_probs(sequences, m, pseudocount=0.5)

for seq in sequences:
    score = score_log_likelihood(seq, m, probs)
    print(score)