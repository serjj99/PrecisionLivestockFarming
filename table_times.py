import pandas as pd


df = pd.read_csv('Dates/Tag_Data.csv', parse_dates=['START_TIME_LOCAL'], index_col='START_TIME_LOCAL')

df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([12], 11)


#max time= '2021-07-01 01:59:00'
df['DAY_OF_WEEK'] = df.index.dayofweek
df['MINUTES'] = 60 * df.index.hour + df.index.minute
df['DAY'] = df.index.day_of_year
df['MONTH'] = df.index.month
df['YEAR'] = df.index.year

# sns.lineplot(x=df.index, y='STATE_NUMBER', data=df)


df_times = pd.DataFrame()
df_times['DAY_OF_WEEK'] = df['DAY_OF_WEEK']
df_times['MINUTES'] = df['MINUTES']
df_times['DAY'] = df['DAY']
df_times['MONTH'] = df['MONTH']
df_times['YEAR'] = df['YEAR']
df_times['ANIMAL_NUMBER'] = df['ANIMAL_NUMBER']
df_times['GROUP_NUMBER'] = df['GROUP_NUMBER']
df_times['STATE_NUMBER'] = df['STATE_NUMBER']

df_times_1 = df_times[(df_times['GROUP_NUMBER'] == 1)]
df_times_2 = df_times[df_times['GROUP_NUMBER'] == 2]
df_times_3 = df_times[df_times['GROUP_NUMBER'] == 3]
df_times_1_3 = df_times[(df_times['GROUP_NUMBER'] == 1) | (df_times['GROUP_NUMBER'] == 3)]

df_times_1.drop(['GROUP_NUMBER'], axis=1, inplace=True)
df_times_2.drop(['GROUP_NUMBER'], axis=1, inplace=True)
df_times_3.drop(['GROUP_NUMBER'], axis=1, inplace=True)
df_times_1_3.drop(['GROUP_NUMBER'], axis=1, inplace=True)



def floor_agroup(n, min_agroup):
    rest = (n % min_agroup)
    if rest == 0:
        return int(n)
    else:
        return int(n + (min_agroup - rest))


def agroup_date(df, min_agroup): 
    for i in range(1440):
        replace = floor_agroup(i, min_agroup)
        df['MINUTES'] = df['MINUTES'].replace([i], replace)
    animals = df['ANIMAL_NUMBER'].unique()
    for a in animals:
        days = df[df['ANIMAL_NUMBER'] == a]['DAY'].unique()
        for day in days:
            minutes = df[(df['ANIMAL_NUMBER'] == a) & (df['DAY'] == day)]['MINUTES'].unique()
            for m in minutes:
                dates_for_min = df[(df['ANIMAL_NUMBER'] == a) & (df['DAY'] == day) & ((df['MINUTES'] == m))]['MINUTES']
                if len(dates_for_min) > 1:
                    df.drop(dates_for_min[:-1].index, inplace=True)
    return df

min_agroup = 30

# df_times_1 = agroup_date(df_times_1, min_agroup)
# df_times_2 = agroup_date(df_times_2, min_agroup)
# df_times_3 = agroup_date(df_times_3, min_agroup)
# df_times_1_3 = agroup_date(df_times_1_3, min_agroup)


# # df_times_1.to_csv('Dates/states_times_1.csv')
# df_times_1.to_csv('Dates/states_times_1_agrup.csv')

# # df_times_2.to_csv('Dates/states_times_2.csv')
# df_times_2.to_csv('Dates/states_times_2_agrup.csv')

# # df_times_3.to_csv('Dates/states_times_3.csv')
# df_times_3.to_csv('Dates/states_times_3_agrup.csv')

# # df_times_1_3.to_csv('Dates/states_times_1_3.csv')
# df_times_1_3.to_csv('Dates/states_times_1_3_agrup.csv')