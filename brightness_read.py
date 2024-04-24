import pandas as pd

now = pd.Timestamp.now(tz="America/New_York").replace(year=2024, second=0, microsecond=0)

irr = pd.read_hdf("brightness.h5", "data", where=["index=now"])["poa_global"][0]

print(irr)
