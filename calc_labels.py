import constants
import pandas as pd


study = "2"  # "1" or "2"
block_name = "exp_S"  # "baseline", "practice", "exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"


df = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}/Block_Labels.csv")

df_block = df.loc[df["block"] == block_name]
df_block.drop(columns=["block", "Unnamed: 3", "PSE"], inplace=True)
df_block.replace("overestimation", 1, inplace=True)
df_block.replace("underestimation", 0, inplace=True)

df_block.to_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block_name}.csv", index=False)
print(df_block)