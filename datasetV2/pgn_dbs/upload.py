
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="val.db",
    path_in_repo="val.db",
    repo_id="deboradum/chess-positions-large",
    repo_type="dataset"
)
upload_file(
    path_or_fileobj="test.db",
    path_in_repo="test.db",
    repo_id="deboradum/chess-positions-large",
    repo_type="dataset"
)
