import pandas as pd


df = pd.read_csv('Dates/Tag_Data.csv', parse_dates=['START_TIME_LOCAL'], index_col='START_TIME_LOCAL')

df.drop(df[df['STATE_NUMBER'] == 15].index, inplace=True)

df['STATE_NUMBER'] = df['STATE_NUMBER'].replace([12], 11)
df['GROUP_NUMBER'] = df['GROUP_NUMBER'].replace([3], 1)


#max time= '2021-07-01 01:59:00'
df['DAY_OF_WEEK'] = df.index.dayofweek
df['MINUTES'] = 60 * df.index.hour + df.index.minute
df['DAY'] = df.index.day_of_year
df['MONTH'] = df.index.month
df['YEAR'] = df.index.year

# sns.lineplot(x=df.index, y='STATE_NUMBER', data=df)


df_times = pd.DataFrame()
df_times['GROUP_NUMBER'] = df['GROUP_NUMBER']
df_times['ANIMAL_NUMBER'] = df['ANIMAL_NUMBER']
df_times['DAY'] = df['DAY']
df_times['MINUTES'] = df['MINUTES']
# df_times['MONTH'] = df['MONTH']
# df_times['YEAR'] = df['YEAR']
df_times['STATE_NUMBER'] = df['STATE_NUMBER']



def agroup_date_for_min(df, df2): 
    df_minutes = pd.DataFrame(columns = ['GROUP_NUMBER' , 'ANIMAL_NUMBER', 'DAY', 'MINUTES', 'STATE_NUMBER'])
    animals = df['ANIMAL_NUMBER'].unique()
    # for a in [1774, 1828]:
    for a in animals[3:4]:
        print(a)
        for g in [1,2]:
            days = df[(df['ANIMAL_NUMBER'] == a) & (df['GROUP_NUMBER'] == g)]['DAY'].unique()
            if len(days) != 0:
                for day in days:
                    minutes = df[(df['ANIMAL_NUMBER'] == a) & (df['GROUP_NUMBER'] == g) & (df['DAY'] == day)]['MINUTES'].unique()
                    duration_max = df2[(df2['ANIMAL_NUMBER'] == a) & (df2['GROUP_NUMBER'] == g) & (df2['DAY'] == day) & (df2['MINUTES'] == minutes[-1])]['DURATION'][0]
                    m_max = int(min(1440, minutes[-1]+duration_max+1))
                    dates_for_min = -1
                    for m in range(minutes[0], m_max):
                        if m in minutes:
                            dates_for_min = df[(df['ANIMAL_NUMBER'] == a) & (df['GROUP_NUMBER'] == g) & (df['DAY'] == day) & (df['MINUTES'] == m)]
                            df_minutes = pd.concat([df_minutes, dates_for_min], ignore_index=True)
                        else:
                            df_minutes = pd.concat([df_minutes, pd.DataFrame([(g,a,day,m,dates_for_min['STATE_NUMBER'][0])], columns = ['GROUP_NUMBER' , 'ANIMAL_NUMBER', 'DAY', 'MINUTES', 'STATE_NUMBER'])], ignore_index=True)
    return df_minutes



def agroup_date_for_min_separate(df, df2, a, g): 
    # df_minutes = pd.DataFrame(columns = ['GROUP_NUMBER' , 'ANIMAL_NUMBER', 'DAY', 'MINUTES', 'STATE_NUMBER'])
    df_minutes = pd.DataFrame(columns = ['DAY', 'MINUTES', 'STATE_NUMBER'])
    
    days = df[(df['ANIMAL_NUMBER'] == a) & (df['GROUP_NUMBER'] == g)]['DAY'].unique()
    if len(days) != 0:
        day0 = days[0]
        for day in days:
            day_mult = day - (day0)
            minutes = df[(df['ANIMAL_NUMBER'] == a) & (df['GROUP_NUMBER'] == g) & (df['DAY'] == day)]['MINUTES'].unique()
            duration_max = df2[(df2['ANIMAL_NUMBER'] == a) & (df2['GROUP_NUMBER'] == g) & (df2['DAY'] == day) & (df2['MINUTES'] == minutes[-1])]['DURATION'][0]
            m_max = int(min(1440, minutes[-1]+duration_max+1))
            dates_for_min = -1
            for m in range(minutes[0], m_max):
                if m in minutes:
                    dates_for_min = df[(df['ANIMAL_NUMBER'] == a) & (df['GROUP_NUMBER'] == g) & (df['DAY'] == day) & (df['MINUTES'] == m)]
                    df_minutes = pd.concat([df_minutes, pd.DataFrame([(day,int(m + 1440*day_mult),dates_for_min['STATE_NUMBER'][0])], columns = ['DAY', 'MINUTES', 'STATE_NUMBER'])], ignore_index=True)
                else:
                    # df_minutes = pd.concat([df_minutes, pd.DataFrame([(g,a,day,m,dates_for_min['STATE_NUMBER'][0])], columns = ['GROUP_NUMBER' , 'ANIMAL_NUMBER', 'DAY', 'MINUTES', 'STATE_NUMBER'])], ignore_index=True)
                    df_minutes = pd.concat([df_minutes, pd.DataFrame([(day,int(m + 1440*day_mult),dates_for_min['STATE_NUMBER'][0])], columns = ['DAY', 'MINUTES', 'STATE_NUMBER'])], ignore_index=True)

        df_minutes.to_csv('Dates/Dates_times/{}/states_times_animal_{}.csv'.format(g,a))
        
    return df_minutes


# df_complete = agroup_date_for_min(df_times, df)

animals = df['ANIMAL_NUMBER'].unique()

# for a in animals:
for a in [1379,1524,1542,1778,1823]:
    # print('{} / {}'.format(int(a), int(animals[-1])))
    print(a)
    for g in [1,2]:
        agroup_date_for_min_separate(df_times, df, a, g)



# df_complete.to_csv('Dates/states_times_complete.csv')

# df_times.to_csv('Dates/states_times.csv')

# df_times = pd.read_csv('Dates/states_times_time.csv')

# df_complete = pd.DataFrame()
# df_complete['GROUP_NUMBER'] = df_times['GROUP_NUMBER']
# df_complete['ANIMAL_NUMBER'] = df_times['ANIMAL_NUMBER']
# df_complete['MINUTES'] = df_times['MINUTES']
# df_complete['DAY'] = df_times['DAY']
# df_complete['STATE_NUMBER'] = df_times['STATE_NUMBER']

# df_complete.to_csv('Dates/states_times.csv')