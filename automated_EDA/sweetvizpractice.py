import seaborn as sns
import sweetviz as sv
import webbrowser

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Generate the Sweetviz report
my_report = sv.analyze(df)

# Save the report as an HTML file
report_path = "C:/Users/abhin/Desktop/output_sweetviz.html"
my_report.show_html(report_path)

# Open the report automatically in the default web browser
webbrowser.open(report_path)
