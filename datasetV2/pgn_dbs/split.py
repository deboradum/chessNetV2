import sqlite3
import math
from tqdm import tqdm

source_db = 'balanced.db'
table = 'positions'

targets = {
    'train.db': 0.8,
    'val.db': 0.1,
    'test.db': 0.1
}

batch_size = 10000  # rows per batch

# Connect to source DB
src_conn = sqlite3.connect(source_db)
src_cursor = src_conn.cursor()

# Get total rows
src_cursor.execute(f"SELECT COUNT(*) FROM {table}")
total_rows = src_cursor.fetchone()[0]

# Calculate split sizes
sizes = {k: math.floor(v * total_rows) for k, v in targets.items()}
sizes['train.db'] += total_rows - sum(sizes.values())  # fix rounding

# Get table schema
src_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
create_table_sql = src_cursor.fetchone()[0]

# Create target DBs and tables
target_conns = {}
target_cursors = {}
for db_name in targets:
    conn = sqlite3.connect(db_name)
    conn.execute(create_table_sql)
    conn.commit()
    target_conns[db_name] = conn
    target_cursors[db_name] = conn.cursor()

def insert_rows(db_name, rows):
    if not rows:
        return
    placeholders = ','.join(['?'] * len(rows[0]))
    insert_sql = f"INSERT INTO {table} VALUES ({placeholders})"
    target_cursors[db_name].executemany(insert_sql, rows)
    target_conns[db_name].commit()

offset = 0
for db_name in ['train.db', 'val.db', 'test.db']:
    rows_left = sizes[db_name]
    print(f"Copying {rows_left} rows to {db_name}...")
    with tqdm(total=rows_left, unit='rows') as pbar:
        while rows_left > 0:
            limit = min(batch_size, rows_left)
            src_cursor.execute(f"SELECT * FROM {table} LIMIT {limit} OFFSET {offset}")
            batch_rows = src_cursor.fetchall()
            if not batch_rows:
                break
            insert_rows(db_name, batch_rows)
            offset += len(batch_rows)
            rows_left -= len(batch_rows)
            pbar.update(len(batch_rows))

src_conn.close()
for conn in target_conns.values():
    conn.close()

