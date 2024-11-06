import json
import torch
import sqlite3

from datasetGen.constants import BIN_SIZE
from torch.utils.data import IterableDataset


def generator_(db_path, batch_size):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ascii_codes, active_bin_128, active_bin_64, win_perc FROM positions"
    )

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows

    conn.close()


class BuildinWinsDataset(IterableDataset):
    def __init__(self, db_path, batch_size):
        self.db_path = db_path
        self.batch_size = batch_size

    def __iter__(self):
        generator = generator_(self.db_path, self.batch_size)
        for batch in generator:
            # Process each row in the batch
            features, labels = [], []
            for row in batch:
                if BIN_SIZE == 128:
                    features.append(torch.tensor(json.loads(row[0])))
                    labels.append(torch.tensor(row[1]))
                elif BIN_SIZE == 64:
                    features.append(torch.tensor(json.loads(row[0])))
                    labels.append(torch.tensor(row[2]))
                elif BIN_SIZE == 1:
                    features.append(torch.tensor(json.loads(row[0])))
                    labels.append(torch.unsqueeze(torch.tensor(row[3]), 0))
                else:
                    raise Exception("Unimplemented bin size")

            yield torch.stack(features), torch.stack(labels)
