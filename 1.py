# Setup
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_mpg = None
try:
    from plotnine.data import mpg as df_mpg
except Exception:
    pass

sns = None
try:
    import seaborn as sns
except Exception:
    pass

# input sample dataset 
data1 = None
try:
    from vega_datasets import data as data1
except Exception:
    pass

# arithmetic
diameter = 88
radius = diameter / 2
height = 72

circumference = 2 * math.pi * radius
area = math.pi * radius**2
volume = area * height

# arithmetic output
circumference, area, volume

# sample variance
def load_pic():
    if df_mpg is not None:
        return df_mpg.copy()
    if sns is not None:
        df = sns.load_dataset('mpg').copy()
        if 'displacement' in df.columns and 'displ' not in df.columns:
            df['displ'] = df['displacement']
        if 'cty' not in df.columns and 'mpg' in df.columns:
            df['cty'] = df['mpg']
        # Create 'class' proxy if missing
        if 'class' not in df.columns:
            df['class'] = df.get('origin', pd.Series(['unknown'] * len(df)))
        return df
    raise RuntimeError("Error.")

mpg_df = load_mpg()
displ = mpg_df['displ'].dropna().to_numpy()

n = len(displ)
sample_mean = displ.sum() / n
diff = displ - sample_mean
sample_variance = (diff**2).sum() / (n - 1)

# show result
sample_mean, sample_variance


# basic summaries
def basic_summary():
    if data1 is not None:
        try:
            url = data1.url('starwars.json')
            return pd.read_json(url)
        except Exception:
            pass
    # Direct URL 
    url = "https://raw.githubusercontent.com/vega/vega-datasets/master/data/starwars.json"
    return pd.read_json(url)

starwars = basic_summary()

n_characters = len(starwars)
species_dtype = starwars['species'].dtype if 'species' in starwars.columns else 'unknown'
na_meaning = "NA = missing/unknown data."
n_tatooine = len(starwars.loc[starwars['homeworld'] == 'Tatooine'])

humans = starwars.loc[starwars['species'] == 'Human'].copy()
humans['birth_year_num'] = pd.to_numeric(humans['birth_year'], errors='coerce')
mean_age_human = humans['birth_year_num'].mean()

print("How many characters are listed?", n_characters)
print("Dtype of `species` column:", species_dtype)
print("Meaning of NA in `hair_color`:", na_meaning)
print("How many characters are from Tatooine?", n_tatooine)
print("For humans, average (mean) birth_year:", mean_age_human)


# some plotting, visualization
plt.figure()
plt.scatter(pd.to_numeric(starwars['height'], errors='coerce'),
            pd.to_numeric(starwars['mass'], errors='coerce'))
plt.xlabel('Height')
plt.ylabel('Mass')
plt.title('Starwars: Height vs Mass')
plt.show()

# Identify heaviest character 
mass_num = pd.to_numeric(starwars['mass'], errors='coerce')
idx_max = mass_num.idxmax()
starwars.loc[idx_max, ['name', 'height', 'mass', 'species', 'homeworld']]


# horizontal boxplot of cty by class with red outliers
df2 = mpg_df[['cty', 'class']].dropna().copy()

def iqr_bounds(values): # Compute IQR 
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

bounds = df2.groupby('class')['cty'].apply(iqr_bounds).to_dict()
df2['is_outlier'] = df2.apply(
    lambda r: (r['cty'] < bounds[r['class']][0]) or (r['cty'] > bounds[r['class']][1]),
    axis=1
)

# Draw 
classes = sorted(df2['class'].unique())
data_by_class = [df2.loc[df2['class'] == c, 'cty'].to_numpy() for c in classes]

plt.figure(figsize=(8, 6))
plt.boxplot(data_by_class, vert=False, labels=classes, manage_ticks=True)

for i, c in enumerate(classes, start=1):
    out_x = df2.loc[(df2['class'] == c) & (df2['is_outlier']), 'cty'].to_numpy()
    out_y = np.full_like(out_x, i, dtype=float)
    plt.scatter(out_x, out_y, s=20, c='red')

plt.xlabel('City MPG (cty)')
plt.ylabel('Class')
plt.title('City MPG by Class (Outliers in Red)')
plt.tight_layout()
plt.show()













