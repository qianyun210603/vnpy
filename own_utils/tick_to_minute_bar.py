from vnpy.trader.constant import Interval, Status, Exchange
from vnpy.trader.database import get_database
from vnpy.trader.utility import BarGenerator
from datetime import datetime
from arctic import Arctic


class _OnBar:
    def __init__(self):
        self.bars = []

    def __call__(self, bar):
        self.bars.append(bar)

    def get_bars(self):
        return self.bars


def convert_to_minite_bar(ticks):
    on_bar = _OnBar()
    bg = BarGenerator(on_bar)
    for tick in ticks:
        bg.update_tick(tick)
    return on_bar.get_bars()


if __name__ == '__main__':
    store = Arctic('localhost')
    database = get_database()
    tslib_tick = store.get_library('tick_data')
    db_symbols = tslib_tick.list_symbols()
    for ds in db_symbols:
        symbol, exch_str = ds.split('_')
        exch = Exchange(exch_str)
        chunks = list(tslib_tick.get_chunk_ranges(ds))
        ticks = database.load_tick_data(
            symbol, exch, datetime.strptime(chunks[0][0].decode(), '%Y-%m-%d %H:%M:%S'),
            datetime.strptime(chunks[-1][1].decode(), '%Y-%m-%d %H:%M:%S.%f'))
        bars = convert_to_minite_bar(ticks)
        database.save_bar_data(bars)
        print(symbol + '.' + exch_str, f": {len(ticks)} ticks converts to {len(bars)} bars")
