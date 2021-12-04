import pandas as pd

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import get_database

import rqdatac


EXCH_MAPPING = {
    'CCFX': Exchange.CFFEX,
    'XINE': Exchange.INE,
    'XSGE': Exchange.SHFE,
    'XZCE': Exchange.CZCE,
    'XDCE': Exchange.DCE,
    'XSHG': Exchange.SSE,
    'XSHE': Exchange.SZSE,
}


if __name__ == "__main__":
    rqdatac.init('15201306382', 'Cyq06284015')
    all_stocks = rqdatac.all_instruments(type='CS').sort_values(by='order_book_id').replace('0000-00-00', '2100-01-01')

    num_klines = 0

    db_mgr = get_database()
    interval = Interval.MINUTE
    assert interval in (Interval.MINUTE, Interval.DAILY), "only 1min or 1 day bar"
    myidx = list(all_stocks.trading_code).index('002292') + 1
    idx = myidx
    for _, row in all_stocks.iloc[myidx:, :].iterrows():
        try:
            symbol, exch_str = row.order_book_id.split('.')
            exch = EXCH_MAPPING[exch_str]
            if row.listed_date.startswith('2999'):
                print(f"{idx}: 0 bars for {symbol}.{exch.name} saved")
                idx += 1
                continue
            if row.de_listed_date.startswith('2999'):
                row.de_listed_date = row.de_listed_date.replace("2999", "2100")
            start_date = max(pd.Timestamp(row.listed_date), pd.Timestamp(year=2014, month=1, day=1))
            end_date = min(pd.Timestamp(row.de_listed_date), pd.Timestamp.now())
            if end_date <= start_date:
                print(f"{idx}: 0 bars for {symbol}.{exch.name} saved")
                idx += 1
                continue
            df_price = rqdatac.get_price(
                order_book_ids=row.order_book_id,
                start_date=start_date,
                end_date=end_date,
                frequency='1m' if interval == Interval.MINUTE else '1d',
                adjust_type='none'
            )
            df_one_name = df_price.loc[row.order_book_id]
            df_one_name.index -= pd.Timedelta(minutes=1)

            bars = [
                BarData(
                    symbol=symbol,
                    exchange=exch,
                    datetime=dt.to_pydatetime(),
                    interval=interval,
                    volume=row.volume,
                    turnover=row.total_turnover,
                    open_interest=row.get('open_interest', 0),
                    open_price=row.open,
                    high_price=row.high,
                    low_price=row.low,
                    close_price=row.close,
                    gateway_name='DB'
                ) for dt, row in df_one_name.tz_localize('Asia/Shanghai').iterrows()
            ]
            db_mgr.save_bar_data(bars)
            num_klines += len(bars)
            print(f"{idx}: {len(bars)} bars for {symbol}.{exch.name} saved")
            idx += 1
        except Exception:
            print(f"in total {num_klines} bars saved")
            raise
    print(f"in total {num_klines} bars saved, final_idx {idx}")




