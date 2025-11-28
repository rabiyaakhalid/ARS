import csv

with open("log.txt") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

for row in rows:
    print(
        "Iter",
        row["Iteration"],
        "steps",
        row["timesteps"],
        "avg reward",
        row["AverageReward"],
    )
