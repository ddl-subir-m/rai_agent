import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Feature Engineering Function
def compute_features(df):
    df['day_of_week'] = df['revenue_recognition_date'].dt.dayofweek
    df['day_of_month'] = df['revenue_recognition_date'].dt.day
    df['month'] = df['revenue_recognition_date'].dt.month

    def latest_same_day_of_week_intramonth(df):
        latest_values = []
        for index, row in df.iterrows():
            same_day_of_week = df[(df['day_of_week'] == row['day_of_week']) & (
                    df['revenue_recognition_date'] < row['revenue_recognition_date']) & (
                                          df['revenue_recognition_date'].dt.month == row[
                                      'revenue_recognition_date'].month)]
            if same_day_of_week.empty:
                latest_values.append(None)
            else:
                latest_values.append(same_day_of_week['day_sales_usd'].values[-1])
        return latest_values

    df['latest_same_day_of_week_intramonth'] = latest_same_day_of_week_intramonth(df)

    def rolling_mean_same_day_of_week(df, n_days=28, use_last_day_prev_month=False):
        rolling_means = []

        for index, row in df.iterrows():
            if use_last_day_prev_month:
                # Find the last day of the previous month
                current_date = row['revenue_recognition_date']
                first_day_of_current_month = pd.Timestamp(current_date.year, current_date.month, 1)
                last_day_of_prev_month = first_day_of_current_month - pd.Timedelta(days=1)

                # Determine the start date for the n_days look-back period
                start_date = last_day_of_prev_month - pd.Timedelta(days=n_days)

                # Filter the DataFrame for the n_days period before the last day of the previous month, same day of
                # the week
                same_period = df[(df['day_of_week'] == row['day_of_week']) &
                                 (df['revenue_recognition_date'] <= last_day_of_prev_month) &
                                 (df['revenue_recognition_date'] > start_date)]

                # Compute the mean for 'day_sales_usd' within the same period
                if same_period.empty:
                    rolling_means.append(None)
                else:
                    rolling_means.append(same_period['day_sales_usd'].mean())
            else:
                # Original logic: Same day of the week within the past n_days
                same_day_of_week = df[(df['day_of_week'] == row['day_of_week']) &
                                      (df['revenue_recognition_date'] < row['revenue_recognition_date']) &
                                      (df['revenue_recognition_date'] >= row['revenue_recognition_date'] - pd.Timedelta(
                                          days=n_days))]

                if same_day_of_week.empty:
                    rolling_means.append(None)
                else:
                    rolling_means.append(same_day_of_week['day_sales_usd'].mean())

        return rolling_means

    df['rolling_mean_same_day_of_week_28_prev_month'] = rolling_mean_same_day_of_week(df, n_days=28,
                                                                                      use_last_day_prev_month=True)
    df['rolling_mean_same_day_of_week_84_prev_month'] = rolling_mean_same_day_of_week(df, n_days=84,
                                                                                      use_last_day_prev_month=True)

    def week_of_month(date):
        first_day = date.replace(day=1)
        dom = date.day
        adjusted_dom = dom + first_day.weekday()
        return (adjusted_dom - 1) // 7 + 1

    df['week_of_month'] = df['revenue_recognition_date'].apply(week_of_month)
    df['week_of_month'] = df['week_of_month'].map({1: 'Week 1', 2: 'Week 2', 3: 'Week 3', 4: 'Week 4', 5: 'Week 5'})

    def part_of_month(day):
        if day <= 10:
            return 'Early'
        elif day <= 20:
            return 'Mid'
        else:
            return 'Late'

    df['part_of_month'] = df['revenue_recognition_date'].dt.day.apply(part_of_month)
    df['part_of_month'] = pd.Categorical(df['part_of_month'], categories=['Early', 'Mid', 'Late'], ordered=True)

    df['first_week'] = df['revenue_recognition_date'].dt.day <= 7
    df['last_week'] = df['revenue_recognition_date'].apply(lambda x: x.day > (pd.Period(x, freq='M').days_in_month - 7))

    df['day_of_month'] = df['revenue_recognition_date'].dt.day
    df['days_in_month'] = df['revenue_recognition_date'].apply(lambda x: pd.Period(x, freq='M').days_in_month)
    df['monthly_sin'] = np.sin(2 * np.pi * df['day_of_month'] / df['days_in_month'])
    df['monthly_cos'] = np.cos(2 * np.pi * df['day_of_month'] / df['days_in_month'])

    df['quarter'] = df['revenue_recognition_date'].dt.quarter
    df['quarter'] = df['quarter'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})

    def start_of_quarter(date):
        return pd.Timestamp(year=date.year, month=(date.quarter - 1) * 3 + 1, day=1)

    def end_of_quarter(date):
        if date.quarter == 4:
            return pd.Timestamp(year=date.year + 1, month=1, day=1)
        else:
            return pd.Timestamp(year=date.year, month=date.quarter * 3 + 1, day=1)

    df['start_of_quarter'] = df['revenue_recognition_date'].apply(start_of_quarter)
    df['days_since_start_of_quarter'] = (df['revenue_recognition_date'] - df['start_of_quarter']).dt.days

    df['end_of_quarter'] = df['revenue_recognition_date'].apply(end_of_quarter)
    df['days_until_end_of_quarter'] = (df['end_of_quarter'] - df['revenue_recognition_date']).dt.days

    df['days_in_quarter'] = df['days_since_start_of_quarter'] + df['days_until_end_of_quarter']
    df['quarterly_sin'] = np.sin(2 * np.pi * df['days_since_start_of_quarter'] / df['days_in_quarter'])
    df['quarterly_cos'] = np.cos(2 * np.pi * df['days_since_start_of_quarter'] / df['days_in_quarter'])

    df.drop(columns=['start_of_quarter', 'end_of_quarter'], inplace=True)

    return df


if __name__ == "__main__":
    series_id = "Atlantis_IrisInnovations"

    # Load your dataset
    data_path = f'/mnt/code/data/{series_id}.csv'
    data = pd.read_csv(data_path)

    # Parse dates
    data['revenue_recognition_date'] = pd.to_datetime(data['revenue_recognition_date'])

    # Sort the data by date
    data = data.sort_values(by='revenue_recognition_date')

    # Compute features
    data = compute_features(data)

    # Filter to the last 45 days
    last_30_days_data = data.tail(45)

    # Plot the day_sales_usd and the rolling means
    plt.figure(figsize=(14, 7))
    plt.plot(last_30_days_data['revenue_recognition_date'], last_30_days_data['day_sales_usd'], label='Day Sales USD',
             color='blue')
    plt.plot(last_30_days_data['revenue_recognition_date'],
             last_30_days_data['rolling_mean_same_day_of_week_28_prev_month'],
             label='28-day Rolling Mean (Same Day of Week)', color='orange')
    plt.plot(last_30_days_data['revenue_recognition_date'],
             last_30_days_data['rolling_mean_same_day_of_week_84_prev_month'],
             label='84-day Rolling Mean (Same Day of Week)', color='green')
    plt.xlabel('Date')
    plt.ylabel('Sales USD')
    plt.title('Day Sales USD and Rolling Means')
    plt.legend()
    plt.show()

    # Display the first few rows to verify the new features
    # print(data.head(20))

    # Save the processed features for further use
    data.to_csv(f'/mnt/code/data/feature_engineering/{series_id}_processed_features.csv', index=False)
    print(f"Feature engineering completed and saved to /mnt/code/data/feature_engineering/{series_id}_processed_features.csv.")