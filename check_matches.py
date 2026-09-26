import sys

path = sys.argv[1] if len(sys.argv) > 1 else "output/matching_results.tsv"
total = 0
nonempty = 0
total_matches = 0
with open(path, encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        total += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 1 and parts[1].strip():
            nonempty += 1
            total_matches += len(parts[1].split(","))

print("total_rows", total)
print("nonempty_rows", nonempty)
print("total_match_ids", total_matches)
