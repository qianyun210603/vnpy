from jqdatasdk import *
import pandas as pd
from datetime import datetime
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData


EXCH_MAPPING = {
    'CCFX': Exchange.CFFEX,
    'XINE': Exchange.INE,
    'XSGE': Exchange.SHFE,
    'XZCE': Exchange.CZCE,
    'XDCE': Exchange.DCE,
    'XSHG': Exchange.SSE,
}


EXCH_MAPPING_REVERSE = {v: k for k, v in EXCH_MAPPING.items()}


INTERVAL_MAPPING_REVERSE = {
    Interval.MINUTE: '1m',
    Interval.HOUR: '60m',
    Interval.DAILY: '1d',
    Interval.WEEKLY: '1w'
}


INTERVAL_MAPPING = {v: k for k, v in INTERVAL_MAPPING_REVERSE.items()}


time_shift = pd.Timedelta('1min')


def get_prices(jq_symbol, start_date, end_date, bar_freq='1m', chunk_size: int = 10000,
               fields=('date', 'open', 'close', 'high', 'low', 'volume', 'money')):
    allchunks = []
    end_time = end_date
    while start_date <= end_time:
        part_data = get_bars(
            jq_symbol, count=chunk_size, unit=bar_freq, end_dt=end_time,
            fields=fields
        ).rename(columns={'money': 'amount'})
        part_data.date = pd.to_datetime(part_data.date)
        part_data.set_index('date', inplace=True)
        if part_data is None or part_data.empty:
            break
        allchunks.append(part_data)
        end_time = part_data.index.min()

    if not bool(allchunks):
        return pd.DataFrame(columns=['date', 'open', 'close', 'high', 'low', 'volume', 'amount'])

    data = pd.concat(allchunks[::-1]) if len(allchunks) > 1 else allchunks[0]
    return data.loc[start_date:]


def save_one_symbol(
        jq_symbol, start_date, end_date, fields=('date', 'open', 'close', 'high', 'low', 'volume', 'money')
):
    symbol, exch_str = jq_symbol.split('.')
    exch = EXCH_MAPPING[exch_str]

    df = get_prices(jq_symbol, start_date, end_date+pd.Timedelta(hours=23), fields=fields)
    if not df.empty:
        bar_lists = [
            # noinspection PyTypeChecker
            BarData(
                symbol=symbol,
                exchange=exch,
                datetime=(dt - time_shift).to_pydatetime(),
                interval=Interval.MINUTE,
                volume=row.volume,
                turnover=row.amount,
                open_interest=row.get('open_interest', 0),
                open_price=row.open,
                high_price=row.high,
                low_price=row.low,
                close_price=row.close,
                gateway_name='DB'
            ) for dt, row in df.tz_localize('Asia/Shanghai').iterrows()
        ]
        database_manager.save_bar_data(bar_lists)
        print(f"{jq_symbol} finished, {len(bar_lists)} records saved")


if __name__ == '__main__':
    auth("13842586876", "Jqdata06284015")

    database_manager = get_database()

    for jq_symbol in ('000016.XSHG', '000300.XSHG', '000905.XSHG'):
        save_one_symbol(jq_symbol, datetime(2014,1,1), datetime.now())

    all_future_contract = get_all_securities(types=['futures'], date=None)
    my_contracts = all_future_contract[
        all_future_contract.name.str.startswith('IF') | all_future_contract.name.str.startswith(
            'IH') | all_future_contract.name.str.startswith('IC')]

    for jq_symbol, row in my_contracts.iterrows():

        assert jq_symbol.endswith('CCFX'), "Only checked for CFFEX for now"
        # noinspection PyTypeChecker
        if row.end_date < pd.Timestamp("2019-01-01"):
            continue
        save_one_symbol(
            jq_symbol, row.start_date, row.end_date,
            fields=('date', 'open', 'close', 'high', 'low', 'volume', 'money', 'open_interest')
        )




