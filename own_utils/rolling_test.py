from datetime import datetime
from vnpy.app.portfolio_strategy import BacktestingEngine
from vnpy.trader.constant import Interval
from vnpy.app.portfolio_strategy.strategies.rolling_contract_strategy1 import BackwardationRollingStrategy

# IF2111.CFFEX,IF2112.CFFEX,IF2203.CFFEX,IF2206.CFFEX

if __name__ == '__main__':
    engine = BacktestingEngine()
    vt_symbols = [
        '000300.SSE', 'IF2012.CFFEX', 'IF2101.CFFEX', 'IF2102.CFFEX', 'IF2103.CFFEX', 'IF2104.CFFEX', 'IF2105.CFFEX', 'IF2106.CFFEX',
        'IF2107.CFFEX', 'IF2108.CFFEX', 'IF2109.CFFEX', 'IF2110.CFFEX', 'IF2111.CFFEX', 'IF2112.CFFEX', 'IF2203.CFFEX'
    ]
    # vt_symbols = [
    #     '000300.SSE', 'IF2012.CFFEX', 'IF2101.CFFEX', 'IF2102.CFFEX', 'IF2103.CFFEX', 'IF2104.CFFEX', 'IF2106.CFFEX',
    #     'IF2109.CFFEX'
    # ]

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.TICK,
        intervals={'000300.SSE': Interval.TICK},
        start=datetime(2020, 12, 17),
        end=datetime(2021, 1, 30),
        rates={x: 0.23 / 10000 for x in vt_symbols},
        slippages={x: 0 for x in vt_symbols},
        sizes={x: 300 for x in vt_symbols},
        priceticks={x: 0.2 for x in vt_symbols},
        capital=1000000,
    )

    setting = {
        "boll_window": 1200,
        "boll_multi_m": 1,
        "boll_multi_fm": 3,
        "boll_multi_q": 4,
        "target_position": 1,
        "abandon_date": 11,
    }
    engine.add_strategy(BackwardationRollingStrategy, setting)
    engine.load_data()
    engine.run_backtesting()
    df = engine.calculate_result()
    engine.calculate_statistics()
    # print(df[["trading_pnl", "holding_pnl", "total_pnl"]])
