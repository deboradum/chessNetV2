# Balances a chess positions database such that there are an equal amount of
# positions for each win probability bin.

import sqlite3
from tqdm import tqdm

bucket_size = 0.05
input_db = "all.db"
output_db = "balanced.db"
num_buckets = int(1 / bucket_size)

input_conn = sqlite3.connect(input_db)
output_conn = sqlite3.connect(output_db)

# Create empty output table with same schema
output_conn.execute("DROP TABLE IF EXISTS positions")
schema = input_conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='positions'"
).fetchone()[0]
output_conn.execute(schema)

bucket_limits = [(i * bucket_size, (i + 1) * bucket_size) for i in range(num_buckets)]

# First pass: count per bucket with progress bar
bucket_counts = []
for low, high in tqdm(bucket_limits, desc="Counting per bucket", unit="bucket"):
    count = input_conn.execute(
        "SELECT COUNT(*) FROM positions WHERE stockfish_win_perc_20 >= ? AND stockfish_win_perc_20 < ?",
        (low, high),
    ).fetchone()[0]
    bucket_counts.append(count)

print("Bucket counts:", bucket_counts)
min_count = min(bucket_counts)
print("Min bucket count:", min_count, "total number of new positions:", min_count*num_buckets)

# Second pass: sample min_count rows per bucket and insert into output database
for (low, high), count in tqdm(
    zip(bucket_limits, bucket_counts),
    total=len(bucket_limits),
    desc="Processing Buckets",
):
    if count == 0:
        continue

    rows = input_conn.execute(
        f"""
        SELECT * FROM positions
        WHERE stockfish_win_perc_20 >= ? AND stockfish_win_perc_20 < ?
        ORDER BY RANDOM() LIMIT ?
        """,
        (low, high, min_count),
    ).fetchall()

    if rows:
        # Insert the sampled rows into the output table
        output_conn.executemany(
            f"INSERT INTO positions VALUES ({','.join(['?'] * len(rows[0]))})", rows
        )

output_conn.commit()
input_conn.close()
output_conn.close()
