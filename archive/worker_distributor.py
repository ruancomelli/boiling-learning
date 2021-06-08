import time

url_path = boiling_learning_path / '_tmp' / 'url.txt'

sleep_time = 10
while not url_path.is_file():
    print('File not found:', url_path)
    print(f'Sleeping for {sleep_time}s')
    time.sleep(sleep_time)

url = url_path.read_text()
print('Connected to sequence distributor at', f'<{url}>')

sequence_distributor = bl.utils.worker.SequenceDistributorClient(url)

for x in sequence_distributor.consume('my_test3', range(10)):
    print('Processing', x)
    time.sleep(20)
