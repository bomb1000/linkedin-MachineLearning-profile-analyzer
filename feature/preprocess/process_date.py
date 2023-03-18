from datetime import date
from datetime import datetime

import dateparser
import pandas as pd
from datetimerange import DateTimeRange


def get_time_range(dates_str):
    """
    Given a dates string, return a DateTimeRange object represents the date time range.

    :param dates_str: dates string like 'Employed\nMay 2019 – Present'
    :return: a DateTimeRange object represents the date time range, None if no input is given
    """
    if dates_str == 'None' or dates_str is None:
        return None
    else:
        try:
            dates = dates_str.replace("Dates Employed\n", "").split(' – ')
            start_date = dates[0]
            end_date = dates[1]

            return DateTimeRange(parse_date(start_date), parse_date(end_date))
        except Exception:
            return None


def parse_date(date_str):
    """
    Parse date string to a date time object, if 'Present' is passed, return current date time
    This will return the first day of the month.

    :param date_str: a date string
    :return: class:`datetime <datetime.datetime>` representing parsed date if successful,
    """
    if date_str == "Present":
        return datetime.now().replace(day=1)
    else:
        return dateparser.parse(date_str).replace(day=1)


def get_year_names(start_date, end_date, with_month=True, prefix=""):
    """
    Given a start_date and end_date return the year names like '1991 Jan' in the duration.

    :param start_date: start date
    :param end_date: end date
    :param with_month: if true, the month part will be output
    :param prefix: prefix to append in the start of the name
    :return: list of year names
    """
    year_names = []
    if with_month:
        year_names.extend(pd.date_range(start_date, end_date, freq='YS').strftime("%Y-%b").tolist())
    else:
        year_names.extend(pd.date_range(start_date, end_date, freq='YS').strftime("%Y").tolist())
    return list(map(lambda name: prefix + name, year_names))


def generate_year_ranges_between_two_dates(start_date, end_date):
    """
    Generate year ranges between two given dates

    :param start_date: start date
    :param end_date: end date
    :return: list of DateTimeRange object (1 year interval) in the duration of start date and end date
    """
    date_str_list = get_year_names(start_date, end_date)

    dates = list(map(lambda date_str: parse_date(date_str), date_str_list))
    result = []
    for i in range(0, len(dates)):
        start = dates[i]
        end = start.replace(month=12, day=31)
        dt_range = DateTimeRange(start, end)
        result.append(dt_range)

    return result


def generate_on_job_status_row(date_range, start_date=date(1980, 1, 1), end_date=date(2020, 1, 1), prefix=""):
    """
    Generate an one-row dataframe with on job status base on years

    :param date_range: a date range
    :param start_date: start date of on job years
    :param end_date: end date of on job years
    :param prefix: a prefix to the field names
    :return: an one-row dataframe with on job status base on years
    """
    year_names = get_year_names(start_date, end_date, with_month=False, prefix=prefix)
    date_range = get_time_range(date_range)
    year_range_list = generate_year_ranges_between_two_dates(start_date, end_date)
    data = {}
    for year_name, year_range in zip(year_names, year_range_list):
        if date_range is None:
            data[year_name] = 0
        else:
            data[year_name] = 1 if year_range.is_intersection(date_range) else 0
    return pd.DataFrame(data, index=[0], columns=year_names)


def generate_on_job_status_df(date_range_df, start_date=date(1980, 1, 1), end_date=date(2020, 1, 1)):
    """
    Given a date range dataframe, return a dataframe that represents candidate on job status base on years

    :param date_range_df: a date range dataframe
    :param start_date: start date of on job years
    :param end_date: end date of on job years
    :return: a dataframe that represents candidate on job status base on years
    """

    field_name_prefix = date_range_df.columns[0] + "/"
    year_names = get_year_names(start_date, end_date, with_month=False, prefix=field_name_prefix)
    on_job_status_df = pd.DataFrame(columns=year_names)
    for index, row in date_range_df.iterrows():
        date_range = row[0]
        new_row = generate_on_job_status_row(date_range, start_date, end_date, prefix=field_name_prefix)
        on_job_status_df = on_job_status_df.append(new_row, ignore_index=True)

    on_job_status_df = on_job_status_df.apply(pd.to_numeric)
    return on_job_status_df
