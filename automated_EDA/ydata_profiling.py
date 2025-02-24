import seaborn as sns
data=sns.load_dataset('tips')

from ydata_profiling import ProfileReport
profile=ProfileReport(data,explorative=True)
profile.to_file('output_ydata_profiling.html')