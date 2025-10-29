import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


df = pd.read_csv("dataset.csv")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(df)


synthetic_data = synthesizer.sample(num_rows=100)

print(synthetic_data.head())
synthetic_data.to_csv("synthetic_bank_data.csv", index=False)
