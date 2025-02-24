import seaborn as sns
import pandas as pd 
# Enable interactive plotting in VS Code
import matplotlib.pyplot as plt
data=sns.load_dataset('tips')
data.to_csv('tips.csv')

from autoviz.AutoViz_Class import AutoViz_Class
AV=AutoViz_Class()
filename=(r"C:\Users\abhin\Desktop\tips.csv")
df = AV.AutoViz(
    filename=filename,   # File path
    sep=',',             # Separator
    depVar='',           # Target variable (if supervised learning)
    dfte=None,           # If you want to pass a dataframe directly, set this
    header=0,            # Row number containing column names
    verbose=2,           # Increase verbosity for more insights
    lowess=False,        # Set True for smooth regression curves
    chart_format='svg',  # Can be 'png', 'svg', etc.
    max_rows_analyzed=50000,  # Limit on rows to analyze
    max_cols_analyzed=30  # Limit on columns to analyze  # Change format to save as PNG
)