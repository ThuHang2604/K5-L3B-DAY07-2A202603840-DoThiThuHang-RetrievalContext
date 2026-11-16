import json

chunks = {c["id"]: c["text"] for c in json.load(open("starter/data/converted/chunks.json", encoding="utf-8"))}
checks = [
    ("B:wal-configuration:001", "5 minutes"),
    ("B:wal-configuration:001", "1 GB"),
    ("B:transaction-iso:011", "first non-transaction-control"),
    ("B:routine-vacuuming:016", "two billion"),
    ("B:wal-async-commit:005", "corruption"),
]
for cid, phrase in checks:
    print(cid, repr(phrase), phrase in chunks[cid])