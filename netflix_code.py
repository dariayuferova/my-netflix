import pandas as pd
import numpy as np
from datetime import timedelta
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

### STEP 1. COLLECTING DATA

### Upload the file:
viewing_activity = pd.read_csv('ViewingActivity.csv')

### Get a first look at the data:
print(viewing_activity.head())
print(viewing_activity.info())

### Check unclear columns:
print(viewing_activity['Attributes'].value_counts()) # leave it, might be interesting:
# "Autoplayed: user action: None" - means that the viewer did not interact with
# that TV show or movie. -> delete rows form the analysis if duration < 1 min
# "Autoplayed: user action: User_Interaction" - means that the viewer interacted with
# the TV show or movie in a browser, by clicking on the video player controls
# or using keyboard shortcuts. -> regular watching activity
#  "Has branched playback" - means that the viewer can make choices during playback,
#  to control what happens next. -> delete rows from the analysis, we can't consider it a show or a movie

print(viewing_activity['Device Type'].value_counts()) # leave it, might be interesting
print(viewing_activity['Supplemental Video Type'].value_counts()) # not interesting
print(viewing_activity['Country'].value_counts()) # could've been interesting but not for us

### STEP 2. CLEANING UP AND PREPARING DATA FOR ANALYSIS

### Delete unnecessary columns and rows leaving only the information regarding tv shows and movies, rename them:
viewing_activity = viewing_activity.drop(list(viewing_activity)[7:10], axis=1)
viewing_activity = viewing_activity.rename(columns={"Profile Name": "profile_name", "Start Time": "start_time",
                                                    "Duration": "duration", "Attributes": "autoplay",
                                                    "Title": "full_title", "Supplemental Video Type": "supp_vid_type",
                                                    "Device Type": "device_type"})
viewing_activity = viewing_activity[viewing_activity.supp_vid_type.isna()] \
    .drop(columns='supp_vid_type') \
    .reset_index(drop=True)

### Clean data based on autoplay column:
viewing_activity = viewing_activity[viewing_activity.autoplay != 'Has branched playback; ']

### Check for duplicates and NAs:
duplicateRows = viewing_activity[viewing_activity.duplicated()]
print(duplicateRows) # no full duplicates
print(viewing_activity.isnull().values.sum()) # 1495 NAs
print(viewing_activity.isnull().values.sum()==len(viewing_activity[viewing_activity.autoplay.isna()]))
# all the NAs account for autoplay column

### Check the users attached to Netflix profile and their activity:
print(viewing_activity['profile_name'].value_counts())

### Delete Дом as it contains information in Russian and is of insignificant number (83 records out of 1847):
viewing_activity = viewing_activity.drop(viewing_activity[viewing_activity.profile_name == 'Дом'].index)

### Rearrange columns so that the device type would be between autoplay and title:
viewing_activity.insert(4, 'device_type', viewing_activity.pop('device_type'))

### Fill the title, season, episode columns with data from column full_title:
viewing_activity['title'] = viewing_activity['full_title']. \
    apply(lambda x: x.split(': ')[0] if 'Episode' in x else x)
viewing_activity['season'] = viewing_activity['full_title']. \
    apply(lambda x: x.split('Season ')[
    1] if 'Episode' in x and 'Season' in x else 'Limited Series' if 'Episode' in x else 'Movie')
viewing_activity['episode'] = viewing_activity['full_title']. \
    apply(lambda x: x.split(' ')[-1] if 'Episode' in x else 'Movie')

### Clean up episode and season columns:
viewing_activity['episode'] = viewing_activity['episode'] \
    .str.replace(')', '', regex=False) \
    .str.replace('One', '1', regex=False)
viewing_activity['season'] = viewing_activity['season'].apply(lambda x: x.split(': ')[0] if ': ' in x else x)

### Add information on movie/series/limited series type of content:
viewing_activity['content_type'] = viewing_activity['season']. \
    apply(lambda x: 'Movie' if x == 'Movie' else 'Limited Series' if x == 'Limited Series' else 'Series')

### Change the type of start_time column to datetime:
viewing_activity['start_time'] = pd.to_datetime(viewing_activity['start_time'])

### Extract month of viewing:
viewing_activity.insert(5, 'month_of_start', viewing_activity['start_time'].dt.strftime('%b'))

### We need to change time records by profile user (Daria - GMT+2/GMT+3 (summertime), Alexander - GMT+3),
### considering Netflix stores data in GMT timezone format:
viewing_activity.insert(2, 'user_start_time', viewing_activity['start_time'] + timedelta(hours=3))
### Create a mask to filter rows that we need to edit:
m = (viewing_activity['profile_name'] == 'Daria') & \
    (viewing_activity['month_of_start'].isin(['Nov', 'Dec', 'Jan', 'Feb', 'Mar']))
viewing_activity.loc[m, 'user_start_time'] = \
    viewing_activity['start_time'] + timedelta(hours=2)

### Extract hour and day of week of viewing:
viewing_activity.insert(3, 'time_of_start', viewing_activity['user_start_time'].dt.strftime('%H'))
viewing_activity['time_of_start'] = pd.to_numeric(viewing_activity['time_of_start'])
viewing_activity.insert(4, 'day_of_week_start', viewing_activity['user_start_time'].dt.strftime('%a'))

### Change duration column datatype to integer:
viewing_activity.insert(7, 'duration_in_mins',
                        pd.to_numeric((pd.to_timedelta(viewing_activity['duration']).dt.total_seconds() / 60))
                        .astype('int64'))

### Check if the values are appropriate and consistent with what we're expecting:
print(viewing_activity['start_time'].min(), viewing_activity['start_time'].max())
print(viewing_activity['duration_in_mins'].min(), viewing_activity['duration_in_mins'].max())

### Check if one title has only one type of content:
consistent_type = viewing_activity[['title', 'content_type']] \
    .groupby('title', observed=True) \
    .nunique('content_type') \
    .reset_index()
consistent_type = consistent_type[consistent_type.content_type != 1]

print(consistent_type)

### Gilmore Girls have both Series with several seasons and Limited Series afterwards
### but with Stranger Things it's a mistake. Clean all the rows with title 'Stranger Things':
m = (viewing_activity['title'] == 'Stranger Things') & (viewing_activity['content_type'] == 'Limited Series')
# created a mask to filter rows that we need to edit
viewing_activity.loc[m, 'season'] = '4'
viewing_activity.loc[viewing_activity.title == 'Stranger Things', 'content_type'] = 'Series'
# check the results:
# print(viewing_activity[viewing_activity.title=='Stranger Things'])

### Let's see what's the percentage of successful autoplay with/without additional user interaction
### (= proportion of views over 1 minute of all 'Autoplayed: user action:' views):
pd.options.mode.chained_assignment = None  # to deal with false positive warnings raising here
autoplay_df = viewing_activity[['autoplay', 'duration_in_mins']]
autoplay_df['success'] = autoplay_df['duration_in_mins'].apply(lambda x: '> 1 min' if x > 1 else '<= 1 min')
autoplay_df = autoplay_df.drop(columns='duration_in_mins') \
    .groupby('autoplay')['success'].value_counts()

# print(autoplay_df)

### Let's remove all entries for less than 1 min for the whole dataset:
viewing_activity_clean = viewing_activity[viewing_activity.duration_in_mins > 1]

### Now remove unnecessary columns:
viewing_activity_clean = viewing_activity_clean.drop(columns=['autoplay', 'start_time', 'duration'])

### Check again for duplicates and NAs after all the modifications:
print(viewing_activity_clean.isnull().values.sum()) # no NAs

duplicateRows_clean = viewing_activity_clean[viewing_activity_clean.duplicated()].reset_index()
# print(duplicateRows_clean) # we have some full duplicates

### Check for partial duplicates based on full title:
duplicateRows_duration = viewing_activity_clean[viewing_activity_clean[['duration_in_mins', 'full_title']].duplicated()] \
    .reset_index()
duplicateRows_time = viewing_activity_clean[viewing_activity_clean[['user_start_time', 'full_title']].duplicated()] \
    .reset_index()
# print(duplicateRows_duration, duplicateRows_time) # we have some partial duplicates

### Remove all the duplicates:
duplicates = pd.merge(duplicateRows_clean, duplicateRows_duration, how='outer')
duplicates = pd.merge(duplicates, duplicateRows_time, how='outer') \
    .set_index('index')
viewing_activity_clean = viewing_activity_clean.drop(index=duplicates.index, axis=0) \
    .reset_index(drop=True)

### We are ready for analysis!

### STEP 3. DESCRIPTIVE ANALYSIS

### Let's get general statistics on views/pieces of content quantitative indicators:

### The average duration of viewing session per user:
duration_stats = viewing_activity_clean.groupby(['profile_name', 'content_type'])['duration_in_mins'].mean() \
    .reset_index() \
    .rename(columns={'duration_in_mins': 'avg_duration_in_mins'})

duration_stats['avg_duration_in_mins'] = duration_stats['avg_duration_in_mins'].astype('int64')

### The number of pieces of content per user:
views_stats = viewing_activity_clean \
    .groupby(['profile_name', 'content_type']) \
    .full_title.agg(['nunique', 'count']) \
    .reset_index()

views_stats = views_stats.rename(columns={'count': 'N_of_views', 'nunique': 'N_of_distinct_pieces'})
views_stats.insert(2, 'N_of_unique_titles', viewing_activity_clean
                   .groupby(['profile_name', 'content_type'])['title'].nunique().reset_index()['title'])
views_stats['views_per_piece'] = round((views_stats['N_of_views'] / views_stats['N_of_distinct_pieces']), 2)

views_df = pd.merge(duration_stats, views_stats, on=['profile_name', 'content_type'], how='outer')

### Sort the table in order that we need for presenting:
views_df = views_df.sort_values(by=['profile_name', 'N_of_unique_titles']).set_index('profile_name')
views_df.insert(5, 'avg_duration_in_mins', views_df.pop('avg_duration_in_mins'))

print(views_df)

### STEP 4. VISUALIZATIONS

### Prepare for visualizations:

### Sort days of the week and months in order that we need for visualizations:
cat_day_of_week = CategoricalDtype(['Mon', 'Tue', 'Wed',
                                    'Thu', 'Fri', 'Sat', 'Sun'], ordered=True)
cat_month = CategoricalDtype(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True)
viewing_activity_clean['day_of_week_start'] = viewing_activity_clean['day_of_week_start'].astype(cat_day_of_week)
viewing_activity_clean['month_of_start'] = viewing_activity_clean['month_of_start'].astype(cat_month)

### Set the overall theme:
my_palette = ["#f5253a", "#620163", "#ba0066", "#fcb292"]
sns.set_theme(style='whitegrid', palette=my_palette)

### Plot the distribution of duration of viewing sessions:
g = sns.FacetGrid(data=viewing_activity_clean, col='profile_name', hue='profile_name', palette=my_palette, height=3.5)
g.map_dataframe(sns.histplot, x='duration_in_mins', binwidth=5, stat='density', kde=True, alpha=0.7)
g.set_axis_labels('Duration, in minutes')
g.set_titles(col_template='{col_name}')
g.fig.suptitle('Distribution of Views by Duration \n\n', weight='bold', y=1.05)

### Let's go a little deeper and see this visualization by duration category:
dur_category = viewing_activity_clean[['profile_name', 'duration_in_mins']]


def categorize_duration(x):
    if x < 30:
        cat = 'less than 30 mins'
    elif 31 < x < 60:
        cat = '31-60 mins'
    else:
        cat = 'more than 60 mins'
    return cat


### Sort categories in order that we need for visualizations:
cat_category = CategoricalDtype(['less than 30 mins', '31-60 mins', 'more than 60 mins'], ordered=True)
dur_category['category'] = dur_category['duration_in_mins'].apply(categorize_duration).astype(cat_category)

dur_category = dur_category.groupby(['profile_name', 'category']) \
    .category.agg(['count']) \
    .reset_index() \
    .rename(columns=({'count': 'number_of_cases'}))
dur_category['sum_per_user'] = dur_category.groupby('profile_name') \
    .number_of_cases.transform('sum')
dur_category['share_of_category'] = round(dur_category['number_of_cases'] / dur_category['sum_per_user'], 2)

fig2, ax = plt.subplots()

sns.barplot(data=dur_category, x='profile_name', y='share_of_category', hue='category') \
    .set(xlabel='',
         ylabel='Share of views')
ax.set_title('Proportion of Viewes by Duration', fontweight='bold')
ax.legend(title='', loc='upper right')
ax.set_yticks(np.arange(0, 1, 0.1))
ax.set_yticklabels([f'{x:.0%}' for x in ax.get_yticks()])  # show y axis with percentages

### Plot the distribution of length of distinct shows:
### We need to filter only Series content_type, count unique episode titles by groups and pivot dataframe to plot:
ax = viewing_activity_clean \
    .drop(viewing_activity_clean[(viewing_activity_clean['season'] == 'Movie') |
                                 (viewing_activity_clean['season'] == 'Limited Series')].index) \
    .groupby(['season', 'profile_name'])['full_title'].nunique() \
    .unstack().plot(kind='bar')
ax.set(xlabel='Season', ylabel='Number of episodes')
ax.set_title('Number of Episodes Viewed by Season per User', weight='bold')
ax.legend(title='')
ax.tick_params(axis='x', rotation=0)

### Let's take a look at the user behaviour:

### Plot what devices users use:

ax = viewing_activity_clean.groupby(['profile_name', 'device_type']).size().unstack().plot(kind='bar', stacked=True)
ax.set_title('Number of Views on Different Devices Per User', fontweight='bold')
ax.set(xlabel='')
ax.tick_params(axis='x', rotation=0)

### Plot the number of views by user:
# by hour:
h = sns.FacetGrid(data=viewing_activity_clean, col='profile_name', height=4,
                  hue='profile_name', palette=my_palette, sharey=False)
h.map_dataframe(sns.histplot, x='time_of_start', binwidth=1, stat='count', alpha=1)
h.set_axis_labels('', 'Number of views')
h.set(xticks=np.arange(0, 24, 1))
h.set_xticklabels(labels=np.arange(0, 24, 1), fontsize=8)  # to fit all the labels
h.set_titles(col_template='{col_name}')
h.fig.suptitle('Distribution of Views by Time of the Day', weight='bold', y=1, fontsize=12)
plt.xlim(0)  # to start labels from the very beginning of axis

# by day of the week:
d = sns.FacetGrid(data=viewing_activity_clean, col='profile_name', height=4,
                  hue='profile_name', palette=my_palette, sharey=False)
d.map_dataframe(sns.histplot, x='day_of_week_start', binwidth=1, stat='count', alpha=1)
d.set_axis_labels('', 'Number of views')
d.set_xticklabels(fontsize=10)
d.set_titles(col_template='{col_name}')
d.fig.suptitle('Distribution of Views by Day of the Week', weight='bold', y=1, fontsize=12)
plt.xlim(-0.5)  # to start labels from the very beginning of axis

# by month:
m = sns.FacetGrid(data=viewing_activity_clean, col='profile_name', height=4,
                  hue='profile_name', palette=my_palette, sharey=False)
m.map_dataframe(sns.histplot, x='month_of_start', binwidth=1, stat='count', alpha=1)
m.set_axis_labels('', 'Number of views')
m.set_xticklabels(fontsize=9)
m.set_titles(col_template='{col_name}')
m.fig.suptitle('Distribution of Views by Month', weight='bold', y=1, fontsize=12)
plt.xlim(-0.5)  # to start labels from the very beginning of axis

### Another way to present views per day/hour is a heatmap:

df = viewing_activity_clean[viewing_activity_clean.profile_name == 'Daria']
df['user_start_time'] = df['user_start_time'].apply(lambda x: x.strftime("%Y-%m-%d, %H:00:00"))
by_hour = df['user_start_time'].value_counts().sort_index(ascending=True)
by_hour.index = pd.to_datetime(by_hour.index)
idx = pd.date_range(min(by_hour.index), max(by_hour.index), freq='1H')
df_count = by_hour.reindex(idx, fill_value=0)
df_count = df_count.rename_axis('datetime').reset_index(name='freq')
df_count['hour'] = df_count['datetime'].dt.hour
df_count['day'] = df_count['datetime'].dt.strftime('%a')
df_count = df_count.drop(['datetime'], axis=1)
df_heatmap = df_count[['day', 'hour', 'freq']].groupby(['day', 'hour']).sum().unstack()
hours_list = list(range(0, 24))
days_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig4, ax = plt.subplots()

ax = sns.heatmap(df_heatmap, linewidth=0.5, yticklabels=days_list, xticklabels=hours_list)
ax.set(xlabel='Hour', ylabel='Day of the week')
ax.set_title('Heatmap of Views Frequency', fontweight='bold')
ax.tick_params(axis='both', rotation=0)

### Check if there is a pattern of binge-watching - only for Series amd Limited Series within one season
### (=start time of the next same title view is within 5 mins from the previous one):

# get only the data we need:
binge_df = viewing_activity_clean.loc[viewing_activity_clean.content_type != 'Movie'].reset_index()

# change the datatype of episode column:
binge_df['episode'] = binge_df['episode'].astype('int64')

# create lags for one view forward:
binge_df['next_start_time'] = binge_df \
    .groupby(['profile_name', 'title'])['user_start_time'].shift(1)
binge_df['next_episode'] = binge_df \
    .groupby(['profile_name', 'title'])['episode'].shift(1)


# create function to iterate through rows to get the ending time:
def get_end_time(row):
    return row['user_start_time'] + timedelta(minutes=row['duration_in_mins'])


# create function to iterate through rows to find binge-watching instances:
def is_binge(row):
    if row['next_start_time'] <= row['end_time'] + timedelta(minutes=5) and row['next_episode'] == row['episode'] + 1:
        return 1
    else:
        return 0


binge_df['end_time'] = binge_df.apply(get_end_time, axis=1)
binge_df['binge'] = binge_df.apply(is_binge, axis=1)
binge_df['share_of_binge'] = binge_df.groupby(['profile_name', 'content_type'])['binge'].transform('sum') / \
                             binge_df.groupby(['profile_name', 'content_type'])['binge'].transform('count')

### Plot the share of binge-watched episodes per user:

ax = binge_df.groupby(['profile_name', 'content_type'])['share_of_binge'].mean().unstack().plot(kind='bar')
ax.set(xlabel='', ylabel='Share of binge-watched episodes')
ax.set_title('Proportion of Binge-Watched Episodes', fontweight='bold')
ax.legend(title='', loc='upper right')
ax.tick_params(axis='x', rotation=0)
ax.set_yticks(np.arange(0, 0.5, 0.1))
ax.set_yticklabels([f'{x:.0%}' for x in ax.get_yticks()])  # show y axis with percentages

### Plot the number of binge-watched episodes by user by time/day of the week:
fig5, ax = plt.subplots(1, 2)

sns.barplot(data=binge_df, x='time_of_start', y='binge', hue='profile_name',
            estimator='sum', errorbar=None, ax=ax[0]) \
    .set(xlabel='',
         ylabel='Number of binge-watched episodes')
ax[0].set_title('Distribution of Binge-Watching by Time of the Day', fontweight='bold')
ax[0].legend(title='', loc='upper left')

sns.barplot(data=binge_df, x='day_of_week_start', y='binge', hue='profile_name',
            estimator='sum', errorbar=None, ax=ax[1]) \
    .set(xlabel='',
         ylabel='Number of binge-watched episodes')
ax[1].set_title('Distribution of Binge-Watching by Day of the Week', fontweight='bold')
ax[1].get_legend().remove()

### Present all the visualizations:
plt.show()
