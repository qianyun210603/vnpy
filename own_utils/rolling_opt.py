from datetime import datetime
from vnpy.app.portfolio_strategy import BacktestingEngine
from vnpy.trader.constant import Interval
from vnpy.app.portfolio_strategy.strategies.rolling_contract_strategy import BackwardationRollingStrategyM
import multiprocessing as mp
import itertools
import pandas as pd
import os


WINDOW_LENGTHS = (240, 480, 720, 1200, 2400)
MS = list(range(1, 2))
QS = list(range(1,6))
FMS = list(range(1, 5))


def evaluate(interval, w, m, q , fm):
    engine = BacktestingEngine()
    vt_symbols = [
        '000300.SSE', 'IF2012.CFFEX', 'IF2101.CFFEX', 'IF2102.CFFEX', 'IF2103.CFFEX', 'IF2104.CFFEX', 'IF2105.CFFEX', 'IF2106.CFFEX',
        'IF2107.CFFEX', 'IF2108.CFFEX', 'IF2109.CFFEX', 'IF2110.CFFEX', 'IF2111.CFFEX', 'IF2112.CFFEX', 'IF2203.CFFEX'
    ]
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=interval,
        intervals={},
        start=datetime(2020, 12, 17),
        end=datetime(2021, 9, 30),
        rates={x: 1.0 / 10000 for x in vt_symbols},
        slippages={x: 0 for x in vt_symbols},
        sizes={x: 300 for x in vt_symbols},
        priceticks={x: 0.2 for x in vt_symbols},
        capital=1000000,
    )
    engine.load_data()
    engine.clear_data()
    setting = {
        "boll_window": w,
        "boll_multi_m": m,
        "boll_multi_fm": fm,
        "boll_multi_q": q,
        "target_position": 1,
        "abandon_date": 11,
    }
    engine.add_strategy(BackwardationRollingStrategyM, setting)
    print(w, fm, m, q)
    engine.run_backtesting()
    df = engine.calculate_result()
    stats = engine.calculate_statistics(df)
    return w, fm, m, q, stats['total_net_pnl'], stats['sharpe_ratio'], stats['max_drawdown'], stats['max_drawdown_duration']


if __name__ == '__main__':

    # vt_symbols = [
    #     '000300.SSE', 'IF2012.CFFEX', 'IF2101.CFFEX', 'IF2102.CFFEX', 'IF2103.CFFEX', 'IF2104.CFFEX', 'IF2106.CFFEX',
    #     'IF2109.CFFEX'
    # ]
    interval = Interval.TICK

    futures = []
    pool = mp.Pool(mp.cpu_count()-6)

    def complete(f):
        g = f.get()
        print(g)
        return g

    for w, m, q, fm in itertools.product(WINDOW_LENGTHS, MS, QS, FMS):
        futures.append(pool.apply_async(evaluate, args=(interval, w, m, q, fm), error_callback=lambda x: print("error:", x)))

    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    resdf = pd.DataFrame([complete(f) for f in futures],
                         columns=['w', 'fm', 'm', 'q', 'total_net_pnl', 'sharpe_ratio', 'max_drawdown', 'max_drawdown_duration'])
    resdf.to_excel(os.path.join(r"D:\Documents\TradeResearch", f"summary_{interval.name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.xlsx"))
