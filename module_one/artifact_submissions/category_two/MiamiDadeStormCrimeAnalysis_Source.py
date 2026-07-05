"""Source export for the DAT 375 Miami-Dade storm and crime analysis.

This file was extracted from the submitted notebook HTML export for CS 499
Module One artifact submission planning. Local MySQL credentials were replaced
with environment-variable based configuration before placing the file in the
artifact_submissions folder.
"""


# %%
import os
import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 220)

# FBI UCR Part I property-crime prefixes on the IBR offense code.
# 200 is arson, 220 is burglary, 23 covers larceny 23A through 23H, 240 is motor vehicle theft.
PROPERTY_PREFIXES = ('200', '220', '23', '240')

# NOAA forecast-zone names that cover Miami-Dade County.
MIAMI_CZ_NAMES = (
    'MIAMI-DADE',
    'COASTAL MIAMI-DADE COUNTY',
    'INLAND MIAMI-DADE',
    'METROPOLITAN MIAMI-DADE',
    'FAR SOUTH MIAMI-DADE COUNTY',
)

# IncidentType labels treated as crimes for this analysis.
# The selection and the borderline-category decisions are explained in the companion report.
CRIME_INCIDENT_TYPES = [
    '18 - HIT AND RUN ACCIDENT',
    '27 - LARCENY THEFT',
    '27V - LARCENY THEFT - MOTOR VEHICLE',
    '27R - LARCENY THEFT – RETAIL',
    '28 - VANDALISM',
    '22 - STOLEN VEHICLE',
    '55M - DOMESTIC VIOLENCE (MISDEMEANOR)',
    '32M - SIMPLE ASSAULT/BATTERY (MISDEMEANOR)',
    '57 - NARCOTICS RELATED INCIDENT OR ARREST',
    '26 - BURGLARY',
    '32F - AGGRAVATED ASSAULT/BATTERY (FELONY)',
    '55F - DOMESTIC VIOLENCE (FELONY)',
    '21 - STOLEN TAG',
    '30SS - SHOTSPOTTER SHOTS FIRED',
    '54 - FRAUD',
    '35 - ALCOHOL RELATED INCIDENT OR ARREST',
    '54 - WORTHLESS DOCUMENT',
    '29 - ROBBERY',
    '26O - BURGLARY - OCCUPIED',
    '32L - ASSAULT/BATTERY ON LEO',
    '16 - DUI',
    '20 - STOLEN DECAL',
    '33FJ - FORCIBLE SEX OFFENSE, JUVENILE',
    '16A - DUI ACCIDENT',
    '29S - ROBBERY - BY SUDDEN SNATCH',
    '58 - PROSTITUTION RELATED INCIDENT OR ARREST',
    '33F - FORCIBLE SEX OFFENSE',
    '33 - SEX OFFENSE',
    '49 - ARSON',
    '31 - HOMICIDE',
    '30 - SHOTS FIRED IN THE AREA',
    '53A - ABDUCTION',
    '26I - BURGLARY - IN PROGRESS',
    '53HT - HUMAN TRAFFICKING',
    '18FA - FATAL HIT AND RUN ACCIDENT',
    '29E - ROBBERY - EXTORTION',
    '47 - BOMB THREAT',
    '48 - EXPLOSION',
]
print(f'Loaded {len(CRIME_INCIDENT_TYPES)} crime IncidentType labels.')

# %%
import os
conn = mysql.connector.connect(
    user=os.getenv('DAT375_MYSQL_USER', 'root'), host=os.getenv('DAT375_MYSQL_HOST', 'localhost'), port='3306',
    password=os.getenv('DAT375_MYSQL_PASSWORD', ''), database=os.getenv('DAT375_MYSQL_DATABASE', 'dat375'),
)
crime_raw = pd.read_sql('SELECT * FROM mpdcrimedata', conn)
storm_raw = pd.read_sql('SELECT * FROM stormevents2024', conn)
conn.close()

print(f'mpdcrimedata:    {crime_raw.shape[0]:,} rows, {crime_raw.shape[1]} columns')
print(f'stormevents2024: {storm_raw.shape[0]:,} rows, {storm_raw.shape[1]} columns')
print()
print('mpdcrimedata columns:   ', list(crime_raw.columns))
print('stormevents2024 columns:', list(storm_raw.columns))

# %%
# Repair the encoding artifact on the 27R label. The source stores the en-dash
# as a 3-character sequence produced by round-tripping UTF-8 bytes through
# Windows-1252; replacing it with a real en-dash lets the filter see those rows.
crime_raw['IncidentType'] = (
    crime_raw['IncidentType'].astype(str).str.replace('â€“', '–', regex=False)
)

# Crime population.
crime = crime_raw[crime_raw['IncidentType'].isin(CRIME_INCIDENT_TYPES)].copy()
print(f'Crime CFS rows: {len(crime):,}')

# Quick look at any curated labels that matched nothing in the source (usually
# a sign of an encoding mismatch like the one above).
hits = crime_raw['IncidentType'].value_counts()
zero_match = [t for t in CRIME_INCIDENT_TYPES if hits.get(t, 0) == 0]
if zero_match:
    print(f'{len(zero_match)} curated label(s) matched zero rows: {zero_match}')

# Highest-volume IncidentTypes not on the curated list, for a quick scan.
off_list = hits.loc[~hits.index.isin(CRIME_INCIDENT_TYPES)].head(15)
print('\nTop 15 IncidentTypes not on the crime list:')
print(off_list.to_string())

# Parse CFSDate so the storm-day comparison uses real datetimes.
crime['CFSDate'] = pd.to_datetime(crime['CFSDate'], errors='coerce')
crime['crime_date'] = crime['CFSDate'].dt.date

# Parse the storm begin/end columns (stored as VARCHAR in DD-MON-YY form).
storm = storm_raw.copy()
storm['begin_dt'] = pd.to_datetime(storm['BEGIN_DATE_TIME'], format='%d-%b-%y %H:%M:%S', errors='coerce')
storm['end_dt']   = pd.to_datetime(storm['END_DATE_TIME'],   format='%d-%b-%y %H:%M:%S', errors='coerce')

# Miami-Dade storm events only.
miami_storms = storm[
    (storm['STATE'].str.upper() == 'FLORIDA')
    & (storm['CZ_NAME'].isin(MIAMI_CZ_NAMES))
].copy()
print(f'\nMiami-Dade storm events: {len(miami_storms):,} '
      f'({miami_storms["EVENT_ID"].nunique():,} unique EVENT_IDs)')

# %%
# A row counts as property if any comma-separated token in IBRCode or FLUCR
# starts with a Part I prefix. The regex anchors on string start or a comma.
prop_pat = r'(?:^|,\s*)(?:' + '|'.join(PROPERTY_PREFIXES) + r')'
crime['is_property'] = (
    crime['IBRCode'].str.contains(prop_pat, regex=True, na=False)
    | crime['FLUCR'].str.contains(prop_pat, regex=True, na=False)
)
n_property = crime['is_property'].sum()
print(f'Property-crime CFS rows: {n_property:,} of {len(crime):,} '
      f'({100 * n_property / len(crime):.2f}%)')

# Storm-day set. Dedup by EVENT_ID so a storm that spans multiple forecast
# zones only contributes its dates once. The begin/end range is expanded
# inclusively at calendar-day resolution.
unique_storms = (
    miami_storms
    .drop_duplicates(subset=['EVENT_ID'])
    .dropna(subset=['begin_dt', 'end_dt'])
)
storm_day_set = set()
for begin, end in zip(unique_storms['begin_dt'], unique_storms['end_dt']):
    storm_day_set.update(pd.date_range(begin.normalize(), end.normalize(), freq='D').date)
print(f'Storm calendar days: {len(storm_day_set):,} (from {len(unique_storms):,} events)')

crime['during_storm'] = crime['crime_date'].isin(storm_day_set)
print(f'Calls on storm days:     {crime["during_storm"].sum():,}')
print(f'Calls on non-storm days: {(~crime["during_storm"]).sum():,}')

# %%
agg = (
    crime.groupby('during_storm')['is_property']
    .agg(['sum', 'count'])
    .rename(index={False: 'Non-storm days', True: 'Storm days'},
            columns={'sum': 'property', 'count': 'cohort'})
)
agg['pct'] = agg['property'] / agg['cohort'] * 100
print(agg.round(2))
lift_pp = agg.loc['Storm days', 'pct'] - agg.loc['Non-storm days', 'pct']
print(f'\nLift (storm minus non-storm): {lift_pp:+.2f} percentage points')

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(agg.index, agg['pct'], color=['#4C72B0', '#C44E52'])
labels = [f'{pct:.2f}%\n({p:,} of {c:,})'
          for pct, p, c in zip(agg['pct'], agg['property'], agg['cohort'])]
ax.bar_label(bars, labels=labels, padding=4, fontsize=11)
ax.set_ylabel('Share of calls that are property crime (%)')
ax.set_title('Property crime share of Miami CFS, storm vs non-storm days (2024)')
ax.set_ylim(0, agg['pct'].max() * 1.25)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig('DAT 375 Project Two_Visualization_VanBoardman.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
combined = crime[[
    'CFSNumber', 'CFSDate', 'crime_date',
    'IncidentType', 'IBRCode', 'FLUCR',
    'is_property', 'during_storm',
    'Neighborhood', 'ZIP Code',
]]
combined.to_csv('DAT 375 Project Two_Visualization_VanBoardman.csv', index=False)
print(f'Wrote {len(combined):,} rows to DAT 375 Project Two_Visualization_VanBoardman.csv')

days_df = pd.DataFrame({'storm_date': sorted(storm_day_set)})
days_df.to_csv('storm_days_miami_2024.csv', index=False)
print(f'Wrote {len(days_df)} storm dates to storm_days_miami_2024.csv')

print('\nSummary metrics')
print('-' * 64)
print(f'Source rows in mpdcrimedata:           {len(crime_raw):,}')
print(f'Crime CFS after IncidentType filter:   {len(crime):,}')
print(f'Miami-Dade storm events (2024):        {len(miami_storms):,}')
print(f'Miami-Dade storm calendar days:        {len(storm_day_set):,}')
print()
print(agg.round(2).to_string())
print(f'\nLift (storm minus non-storm), pp:      {lift_pp:+.2f}')
