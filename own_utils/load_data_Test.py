from vnpy.trader.database import get_database
from vnpy.trader.constant import Direction, Offset, Interval, Status, Exchange
from datetime import datetime
import pandas as pd


def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime
):
    """"""
    database = get_database()

    return database.load_tick_data(
        symbol, exchange, start, end
    )


if __name__ == '__main__':
    datas = load_tick_data('IF2104', Exchange.CFFEX, datetime(2021, 1, 4), datetime(2021, 1, 8))
    cols = ['datetime', 'bid_price_1', 'ask_price_1', 'last_price']
    df = pd.DataFrame([[getattr(td, c) for c in cols] for td in datas], columns=cols).set_index('datetime').sort_index()
    print(df)