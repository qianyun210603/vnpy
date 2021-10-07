import pandas as pd
from jqdatasdk import auth, is_auth, get_all_securities
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np

from vnpy.app.portfolio_strategy import StrategyTemplate, StrategyEngine
from vnpy.trader.utility import BarGenerator
from vnpy.trader.object import TickData, BarData


class BackwardationRollingStrategy(StrategyTemplate):
    """"""

    author = "Booksword"

    band_floor = 3
    boll_window = 480
    boll_dev = 2

    current_spread = 0.0
    boll_mid = 0.0
    boll_down = 0.0
    boll_up = 0.0

    parameters = [
        "band_floor",
        "boll_window",
        "boll_dev",
    ]
    variables = [
        "boll_mid",
        "boll_down",
        "boll_up",
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: List[str],
        setting: dict
    ):
        """"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)

        self.bgs: Dict[str, BarGenerator] = {}
        self.targets: Dict[str, int] = {}
        self.last_tick_time: Optional[datetime] = None

        self.spread_count: int = 0
        self.contracts_same_day = 4
        self.spread_datas: Dict[Tuple[int, int], np.array] = {
            (i, j): np.zeros(self.boll_window) for i in range(self.contracts_same_day)
            for j in range(i+1, self.contracts_same_day)
        }
        self.bounds: Dict[Tuple[int, int], Dict] = {}

        self.parameter_date: Optional[datetime] = None

        self.symbol_mapping: Dict[str, str] = {}
        self.contract_info = get_contract_info('IF')
        self.vt_symbols_today: List[str] = []
        self.expiries: List[datetime] = []

        def on_bar(bar: BarData):
            """"""
            pass

        for vt_symbol in self.vt_symbols:
            self.targets[vt_symbol] = 0
            self.bgs[vt_symbol] = BarGenerator(on_bar)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")
        self.load_bars(5)

    def on_day_open(self, today) -> None:

        df = self.contract_info
        raw = self.contract_info[
                (self.contract_info.start_date <= today) &
                (self.contract_info.end_date + pd.Timedelta(hours=23) >= today)
            ].sort_values(by='end_date').index.to_list()
        self.vt_symbols_today = [x + '.CFFEX' for x in raw]
        self.expiries = [x.to_pydatetime() for x in self.contract_info.loc[raw, 'end_date']]
        print(self.vt_symbols_today)
        print([(x - today.replace(hour=0, minute=0, second=0)).days for x in self.expiries])
        for i in range(self.contracts_same_day):
            for j in range(self.contracts_same_day):
                std = self.spread_datas[(i, j)].std()
                mean = self.spread_datas[(i, j)].mean()
                params = {
                    'mean': mean,
                    'std': std,
                    'bandwidth': max(self.boll_dev * std, self.band_floor)
                }
                self.bounds[(i, j)] = params
                print(today, (i, j), str(params))


    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        if (
            self.last_tick_time
            and self.last_tick_time.minute != tick.datetime.minute
        ):
            bars = {}
            for vt_symbol, bg in self.bgs.items():
                bars[vt_symbol] = bg.generate()
            self.on_bars(bars)

        bg: BarGenerator = self.bgs[tick.vt_symbol]
        bg.update_tick(tick)

        self.last_tick_time = tick.datetime

    def on_bars(self, bars: Dict[str, BarData]):
        """"""
        self.cancel_all()

        for i in range(self.contracts_same_day):
            for j in range(self.contracts_same_day):
                si = self.vt_symbols_today[i]
                sj = self.vt_symbols_today[j]
                current_spread = bars[si].close_price - bars[sj].close_price
                self.spread_datas[(i, j)][:-1] = self.spread_datas[(i, j)][1:]
                self.spread_datas[(i, j)][-1] = current_spread
                # print(f"{bars[si].datetime.isoformat()} spread btw {si} and {sj} is {current_spread:.02f}")

        for idx, vt_symbol in enumerate(self.vt_symbols_today):
            current_pos_this_symbol = self.get_pos(vt_symbol)
            if current_pos_this_symbol > 0:
                tick_sprds = [bars[s].close_price - bars[vt_symbol].close_price for s in self.vt_symbols_today]
                normalized = [(ts-self.bounds[(s, idx)]['mean'])/self.bounds[(i, j)]['bandwidth'] for ]



        self.put_event()


def get_contract_info(und_symbol):
    if not is_auth():
        auth("13842586876", "Jqdata06284015")
    all_future_contract = get_all_securities(types=['futures'], date=None)
    my_contracts = all_future_contract[all_future_contract.name.str.startswith(und_symbol)].copy()
    my_contracts.start_date = my_contracts.start_date.dt.tz_localize('Asia/Shanghai')
    my_contracts.end_date = my_contracts.end_date.dt.tz_localize('Asia/Shanghai')
    return my_contracts.set_index('name').drop(['IF8888', 'IF9999'])