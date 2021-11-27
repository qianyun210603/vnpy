from rqdatac import *
import pandas as pd
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


def get_prices(order_book_id, start_date, end_date, bar_freq='1m', fields=('open', 'close', 'high', 'low', 'volume', 'total_turnover')):
    data = get_price(
        order_book_id, start_date=start_date, end_date=end_date, frequency=bar_freq, fields=fields, adjust_type='none',
        skip_suspended=False, market='cn', expect_df=True
    ).rename(columns={'total_turnover': 'amount'})
    data.index.rename({'datetime': 'date'}, inplace=True)
    return data.loc[order_book_id]


def save_one_symbol(
        symbol, exchange, start_date, end_date, fields=('open', 'close', 'high', 'low', 'volume', 'total_turnover')
):
    order_book_id = symbol if exchange == Exchange.CFFEX else symbol+'.XSHG'
    df = get_prices(order_book_id, start_date, end_date+pd.Timedelta(hours=23), fields=fields)
    if not df.empty:
        bar_lists = [
            # noinspection PyTypeChecker
            BarData(
                symbol=symbol,
                exchange=exchange,
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
        print(f"{symbol}.{exchange.name} finished, {len(bar_lists)} records saved")


if __name__ == '__main__':
    init("15201306382", "Cyq06284015")

    database_manager = get_database()

    # for symbol in ('000016', '000300', '000905'):
    #     save_one_symbol(symbol, Exchange.SSE, datetime(2014,1,1), datetime.now())

    all_future_contract = all_instruments(type='Future', market='cn', date=None)
    my_contracts = all_future_contract[
        all_future_contract.order_book_id.str.startswith('IF') | all_future_contract.order_book_id.str.startswith(
            'IH') | all_future_contract.order_book_id.str.startswith('IC')].copy()
    my_contracts['maturity_date'] = pd.to_datetime(my_contracts['maturity_date'].replace('0000-00-00', '2200-01-01'))
    my_contracts['listed_date'] = pd.to_datetime(my_contracts['listed_date'].replace('0000-00-00', '1970-01-01'))
    my_contracts['de_listed_date'] = pd.to_datetime(my_contracts['de_listed_date'].replace('0000-00-00', '2200-01-01'))
    # noinspection PyTypeChecker
    global_start = pd.Timestamp("2018-01-01")
    for _, row in my_contracts.iterrows():

        if row.de_listed_date < global_start:
            continue
        save_one_symbol(
            row.order_book_id, Exchange(row.exchange), max(global_start, row.listed_date), min(pd.Timestamp.now(), row.de_listed_date),
            fields=('open', 'close', 'high', 'low', 'volume', 'total_turnover', 'open_interest')
        )




