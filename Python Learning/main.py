
# from pathlib import Path
# import pandas as pd

# key = 'Wire Temperature (corrected) [deg C]'
# df = pd.read_csv(Path('.') / 'experiments' / 'Experiment Output 2020-02-14' / 'Experiment 10-25 (0).csv', usecols=[key])
# print(df[key].max())

if __name__ == '__main__':
    import sandbox.mirror
    import sandbox.filters
    import sandbox.sandboxing
