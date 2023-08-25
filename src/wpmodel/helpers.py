from sprucepy import secrets
from azure.storage.blob import BlobServiceClient 

import yaml
import io

from . import constants
    
def get_secrets():
    return {
        k: secrets.get_secret_by_key(k, api_url=constants.SPRUCE_API_URL)
        for k in constants.SECRETS
    }
    
def container_conn( ):
    """Establish Azure cloud connection to a blob container 
    """
    sd =get_secrets()
    container_client = BlobServiceClient(
        account_url=sd['Azure_account_url'],
        credential=sd['Azure_account_key']
    ).get_container_client(container=sd['Azure_ml_blob_container'])
    return container_client

agg_map = {k:[f'pred_{k}',f'actual_{k}'] for k in ['day','week','month','quarter']}

def add_agg_column(df, date_column: str,agg: str
    ):
    '''Add standardized date column based on aggregation level

    Parameters
    ----------
    df: dataframe
        the dataframe to add date column to
    date_column: str
        the day level datetime column of this df
    agg: str
        date aggregation level: day, week, month, quarter
    '''
    if agg == 'day':
        df[agg] =df[date_column].dt.date
    elif agg == 'week':
        df[agg] = df[date_column].dt.to_period('W-SAT').dt.start_time
    elif agg == 'month':
        df[agg] = df[date_column].apply(lambda dt: dt.replace(day=1))
    elif agg == 'quarter':
        df[agg] = df[date_column].apply(lambda x:'{} Q{}'.format(x.year, (x.month - 1) // 3 + 1))
    return df

def now(
    isstr : bool = False,
    fmt : str = '%Y-%m-%dT%H:%M:%S.%f%z',
    tz : str = 'UTC'
):
    """Return today's date as a datetime.datetime object or string

    Parameters
    ----------
    isstr : bool
        whether to return the datetime as a string
    fmt : str
        optional, format if datetime returned as string
    tz : str
        a timezone for the datetime

    Returns
    -------
    str or datetime.datetime
        the current date and time
    """

    now_dt = dt.datetime\
        .utcnow()\
        .replace(tzinfo=pytz.timezone('UTC'))\
        .astimezone(pytz.timezone(tz))

    return date_to_str(now_dt, fmt) if isstr else now_dt


def today(
    isstr : bool = False,
    fmt : str = 'data',
    tz : str = 'UTC'
):
    """Return today's date as a datetime.date object or string

    Parameters
    ----------
    isstr : bool
        whether to return the date as string
    fmt : str
        optional, format if date returned as string
    tz : str
        a timezone for the date

    Returns
    -------
    str or datetime.date
    """

    today_dt = now(tz=tz).date()

    return date_to_str(today_dt, fmt) if isstr else today_dt


def day_offset(
    offset : int = -1,
    isstr : bool = False,
    fmt : str = 'data',
    tz : str = 'UTC'
):
    """Return offset date as a datetime.date object or string

    Parameters
    ----------
    offset : int
        how many days to move ahead or behind
    isstr : bool
        whether to return the date as string
    fmt : str
        optional, format if date returned as string
    tz : str
        a timezone for the date

    Returns
    -------
    str or datetime.date
    """
    today_dt = today(tz=tz)

    offset_dt = today_dt + dt.timedelta(days=offset)

    return date_to_str(offset_dt, fmt) if isstr else offset_dt

def str_to_date(
    date_str : str,
    unix : bool = False
):
    """Parse a date object from a string

    Parameters
    ----------
    date_str : str
        the string to parse
    unix : bool
        should the date be returned as unix time?

    Returns
    -------
    datetime.datetime
        the parsed string
    """

    if date_str is None:
        return

    dttm = parser.parse(date_str)

    if unix:
        return dttm.timestamp() * 1000
    else:
        return dttm

def date_to_str(date, fmt : str = 'pretty_long'):
    """Convert a date or datetime object to a string

    Parameters
    ----------
    date : object with strftime method
        the datetime-like object to convert to str
    fmt : str
        either a key from cfg['date_formats'] or a format string

    Returns
    -------
    str
        the string form of the date following the specified fmt
    """

    if isinstance(date, dt.datetime):
        formats_dict = cfg['time_formats']
    else:
        formats_dict = cfg['date_formats']

    fmt = formats_dict[fmt] if fmt in formats_dict.keys() else fmt

    if isinstance(date, str):
        date = str_to_date(date)

    return date.strftime(fmt)
       
class FakeFile:
    def __init__(self, inbytes):
        self.f = io.BytesIO()
        self.f.write(inbytes)
        self.f.seek(0)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        self.f.close()

