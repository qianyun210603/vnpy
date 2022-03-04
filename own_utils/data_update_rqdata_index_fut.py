import pandas as pd
import logging
from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import get_database

import rqdatac

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', level=logging.INFO,
    filename='rqdata_stock_bar_update.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


EXCH_MAPPING = {
    'INE': Exchange.INE,
    'SHFE': Exchange.SHFE,
    'CZCE': Exchange.CZCE,
    'DCE': Exchange.DCE,
    'XSHG': Exchange.SSE,
    'XSHE': Exchange.SZSE,
    'CFFEX': Exchange.CFFEX
}


if __name__ == "__main__":
    rqdatac.init('15201306382', 'Cyq06284015')
    all_futures = rqdatac.all_instruments(type='Future').replace('0000-00-00', '2100-01-01')

    num_klines = 0

    db_mgr = get_database()
    bar_in_db_last_dates = {(ov.symbol, ov.exchange, ov.interval): ov.end for ov in db_mgr.get_bar_overview()}
    for interval in (Interval.MINUTE, Interval.DAILY):
        myidx = 0 # list(all_stocks.trading_code).index('689009') + 1
        idx = myidx
        for _, row in all_futures.iloc[myidx:, :].iterrows():
            try:
                symbol = row.order_book_id
                exch = EXCH_MAPPING[row.exchange]
                if row.listed_date.startswith('2999'):
                    logging.info(f"{idx}: 0 bars for {symbol}.{exch.name} saved")
                    idx += 1
                    continue
                if row.de_listed_date.startswith('2999'):
                    row.de_listed_date = row.de_listed_date.replace("2999", "2100")
                start_date = max(
                    pd.Timestamp(row.listed_date, tz='Asia/Shanghai'), bar_in_db_last_dates.get((symbol, exch, interval),
                        pd.Timestamp(year=2017, month=1, day=1, tz='Asia/Shanghai')) + pd.Timedelta(minutes=1, seconds=1),
                )
                # print(start_date, bar_in_db_last_dates.get((symbol, exch),
                #         pd.Timestamp(year=2014, month=1, day=1, tz='Asia/Shanghai')))
                end_date = min(pd.Timestamp(row.de_listed_date, tz='Asia/Shanghai'),
                               pd.Timestamp.now(tz='Asia/Shanghai').replace(hour=0, minute=0, second=0, microsecond=0))
                if end_date <= start_date:
                    logging.info(f"{idx}: 0 {interval.name} bars for {symbol}.{exch.name} saved")
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
                logging.info(f"{idx}: {len(bars)} {interval.name} bars for {symbol}.{exch.name} saved (from {df_one_name.index.min().isoformat()})")
                idx += 1
            except Exception:
                logging.error(f"in total {num_klines} bars saved")
                raise
        logging.info(f"in total {num_klines} {interval.name} bars saved, final_idx {idx}")