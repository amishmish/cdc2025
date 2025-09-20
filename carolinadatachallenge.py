import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

realValAdd = pd.read_csv("spaceRealValueAdded.csv", )
realValAdd

employment = pd.read_csv('spaceEmployment.csv')

import pandas as pd
import numpy as np

df_sis = pd.read_csv(r"yearly-number-of-objects-launched-into-outer-space.csv", )

df_sis_cleaned = df_sis[df_sis['Entity'] == "United States"]
df_sis_cleaned = df_sis_cleaned[["Year",  "Annual number of objects launched into outer space"]]
df_sis_cleaned = df_sis_cleaned[df_sis_cleaned["Year"].isin(range(2012, 2024))]
df_sis_cleaned.to_csv("cleaned_space_data.csv", index=False)

print(df_sis_cleaned)

employment.T