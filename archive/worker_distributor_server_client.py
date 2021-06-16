from pathlib import Path

import boiling_learning as bl

boiling_learning_path = Path()

# Server
s = bl.utils.worker.SequenceDistributorServer(
    boiling_learning_path / '_tmp',
    port=5000,
    # venv_name=venv_name
)
s.run()

# Client
c = bl.utils.worker.SequenceDistributorClient.from_file(
    boiling_learning_path / '_tmp' / 'url.txt', sleep_time=10, verbose=True
)
c.connect()
