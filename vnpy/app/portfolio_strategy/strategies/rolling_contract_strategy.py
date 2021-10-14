import os
import os.path
import pickle
import pandas as pd
from jqdatasdk import auth, is_auth, get_all_securities
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dateutil import tz

import numpy as np

from vnpy.app.portfolio_strategy import StrategyTemplate, StrategyEngine
from vnpy.trader.utility import BarGenerator
from vnpy.trader.object import TickData, BarData, TradeData, OrderData
# from vnpy.trader.constant import Status


class BackwardationRollingStrategy(StrategyTemplate):
    """"""

    author = "Booksword"
    price_add = 25
    band_floor = 3
    boll_window = 480
    boll_dev = 2
    target_position = 0

    current_spread = 0.0
    boll_mid = 0.0
    boll_down = 0.0
    boll_up = 0.0

    parameters = [
        "band_floor",
        "boll_window",
        "boll_dev",
        "target_position"
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
        self.last_bar_time: Optional[datetime] = datetime(1970, 1, 1, tzinfo=tz.gettz('Asia/Shanghai'))
        self.minute_bars: Dict[str, BarData] = {}
        self.underlying_symbol = vt_symbols[-1][:2]
        self.ticks: Dict[str, TickData] = {}

        self.spread_count: int = 0
        self.contracts_same_day = 4
        self.spread_datas: Dict[Tuple[int, int], np.array] = {
            (i, j): np.zeros(self.boll_window) for i in range(self.contracts_same_day)
            for j in range(self.contracts_same_day)
        }
        self.bounds: Dict[Tuple[int, int], Dict] = {}

        self.parameter_date: Optional[datetime] = None

        self.symbol_mapping: Dict[str, str] = {}
        self.contract_info = None
        self.vt_symbol_spot = vt_symbols[0]
        self._load_auxiliary_data()
        self.vt_symbols_today: List[str] = []
        self.expiries: List[datetime] = []

        self.switches = {}

        for vt_symbol in self.vt_symbols:
            self.targets[vt_symbol] = 0
            self.bgs[vt_symbol] = BarGenerator(self.on_bar)

    def _load_auxiliary_data(self):
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.__class__.__name__)
        os.makedirs(cache_path, exist_ok=True)
        contract_list_path = os.path.join(cache_path, 'contract.bin')
        contract_list = None
        if os.path.exists(contract_list_path):
            with open(contract_list_path, "rb") as f:
                contract_records = pickle.load(f)
            record_date = contract_records['date']
            contract_list = pd.DataFrame(contract_records['records'])
            if not contract_list[(contract_list.end_date >= record_date) &
                                 (contract_list.end_date <= pd.Timestamp.now())].empty:
                contract_list = None
        if contract_list is None:
            if not is_auth():
                auth("13842586876", "Jqdata06284015")
            contract_list = get_all_securities(types=['futures'], date=None)
            contract_records = {'date': datetime.today(), 'records': contract_list.to_dict()}
            with open(contract_list_path, "wb") as f:
                pickle.dump(contract_records, f)
        my_contracts = contract_list[contract_list.name.str.startswith(self.underlying_symbol)].copy()
        my_contracts.start_date = my_contracts.start_date.dt.tz_localize('Asia/Shanghai')
        my_contracts.end_date = my_contracts.end_date.dt.tz_localize('Asia/Shanghai')
        self.contract_info = my_contracts.set_index('name').drop(['IF8888', 'IF9999'])

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bars(3)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_day_open(self, today) -> None:

        raw = self.contract_info[
                (self.contract_info.start_date <= today) &
                (self.contract_info.end_date + pd.Timedelta(hours=23) >= today)
            ].sort_values(by='end_date').index.to_list()
        self.vt_symbols_today = [x + '.CFFEX' for x in raw]
        self.expiries = [x.to_pydatetime() for x in self.contract_info.loc[raw, 'end_date']]
        # print(self.vt_symbols_today)
        # print([(x - today.replace(hour=0, minute=0, second=0)).days for x in self.expiries])
        for i in range(self.contracts_same_day):
            for j in range(self.contracts_same_day):
                std = self.spread_datas[(i, j)].std()
                mean = self.spread_datas[(i, j)].mean()
                params = {
                    'mean': mean,
                    'std': std,
                    'bandwidth': max(std, self.band_floor)
                }
                self.bounds[(i, j)] = params
                # print(today, (i, j), str(params))
        print(today, self.bounds[(1, 2)])

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.ticks[tick.vt_symbol] = tick
        if (
            self.last_tick_time
            and self.last_tick_time.minute != tick.datetime.minute
        ):

            bg = self.bgs[tick.vt_symbol]
            # print(tick.vt_symbol, tick.datetime)
            self.on_bar(bg.generate())
        # from datetime import time
        # # if tick.datetime.time() > time(14, 59):
        # #     print(tick.datetime)
        bg: BarGenerator = self.bgs[tick.vt_symbol]
        bg.update_tick(tick)

        self.last_tick_time = tick.datetime

        if not all(vt_sym in self.ticks for vt_sym in self.vt_symbols_today) or bool(self.get_all_active_orderids()):
            return

        days_to_expiry_for_0 = (self.expiries[0] - tick.datetime).days
        expiry_penalty_factor = np.exp(0.3 * (5 - min(5, days_to_expiry_for_0 - 2)))
        # total = sum(holdings)
        # print(f"{bar_timestamp} - holdings: {str(holdings)}, total: {total}")
        # if total > self.target_position:
        # print(f"{bar_timestamp} - before order holdings: {str(holdings)}, total: {total}")
        ticks: Dict[str, TickData] = self.ticks
        vt_symbol = tick.vt_symbol
        idx = self.vt_symbols_today.index(vt_symbol)
        current_pos_this_symbol = self.get_pos(vt_symbol)
        if current_pos_this_symbol > 0:
            tick_sprds = [ticks[s].ask_price_1 - ticks[vt_symbol].bid_price_1 for s in self.vt_symbols_today]
            normalized = np.array([(ts - self.bounds[(s, idx)]['mean']) / self.bounds[(s, idx)]['bandwidth']
                                   for s, ts in enumerate(tick_sprds)])
            if days_to_expiry_for_0 < 2:
                argmin = np.argmin(normalized[1:]) + 1
            else:
                normalized[0] /= expiry_penalty_factor
                argmin = np.argmin(normalized)

            if argmin == idx or argmin == 0 or argmin == 3:
                return

            if normalized[argmin] < -self.boll_dev:
                # print(bar_timestamp.isoformat(), normalized, argmin)
                target_price = ticks[vt_symbol].bid_price_1 - self.bounds[(idx, argmin)]['mean'] - \
                               self.bounds[(argmin, idx)]['bandwidth'] * \
                               self.boll_dev * (expiry_penalty_factor if argmin == 0 else 1.0)

                self.buy(self.vt_symbols_today[argmin], target_price, current_pos_this_symbol)
                close_price = ticks[vt_symbol].bid_price_1 - 5
                self.switches.update({self.vt_symbols_today[argmin]: (vt_symbol, close_price)})
                # self.sell(vt_symbol, close_price, current_pos_this_symbol)
                # print(f"{bar_timestamp.isoformat()} - switch from idx {idx} "
                #       f"({vt_symbol}) @ {close_price:.2f} to idx"
                #       f" {argmin} ({self.vt_symbols_today[argmin]} @{target_price:.2f})")

    def on_bar(self, bar: BarData) -> None:
        self.minute_bars[bar.vt_symbol] = bar
        # print(bar.datetime.isoformat(), bar.vt_symbol, bar.open_price, bar.high_price, bar.low_price, bar.close_price)
        if bar.datetime > self.last_bar_time:
            self.last_bar_time = bar.datetime
        # print(self.last_bar_time.isoformat(), bar.vt_symbol, bar.datetime.isoformat(),
        # self.minute_bars[self.vt_symbol_spot].datetime)
        if all(s in self.minute_bars and self.last_bar_time == self.minute_bars[s].datetime
               for s in self.vt_symbols_today) and self.minute_bars[self.vt_symbol_spot].datetime == self.last_bar_time:
            self.on_bars()

    def on_bars(self):
        """"""
        # self.cancel_all()
        bars = self.minute_bars
        bar_timestamp = bars[self.vt_symbols_today[0]].datetime
        for i in range(self.contracts_same_day):
            si = self.vt_symbols_today[i]
            # barsi = self.minute_bars[si]
            # print(barsi.datetime.isoformat(), barsi.vt_symbol, barsi.open_price, barsi.high_price, barsi.low_price,
            #       barsi.close_price)
            for j in range(self.contracts_same_day):
                sj = self.vt_symbols_today[j]
                current_spread = bars[si].close_price - bars[sj].close_price
                self.spread_datas[(i, j)][:-1] = self.spread_datas[(i, j)][1:]
                self.spread_datas[(i, j)][-1] = current_spread

        if not self.trading:
            return
        holdings = [self.get_pos(vt_s) for vt_s in self.vt_symbols_today]
        days_to_expiry_for_0 = (self.expiries[0] - bar_timestamp).days

        if not bool(self.ticks) and bool(self.minute_bars):
            expiry_penalty_factor = np.exp(0.3 * (5 - min(5, days_to_expiry_for_0 - 2)))
            # total = sum(holdings)
            # print(f"{bar_timestamp} - holdings: {str(holdings)}, total: {total}")
            # if total > self.target_position:
            # print(f"{bar_timestamp} - before order holdings: {str(holdings)}, total: {total}")
            for idx, vt_symbol in enumerate(self.vt_symbols_today):
                current_pos_this_symbol = self.get_pos(vt_symbol)
                if current_pos_this_symbol > 0:
                    tick_sprds = [bars[s].close_price - bars[vt_symbol].close_price for s in self.vt_symbols_today]
                    normalized = np.array([(ts-self.bounds[(s, idx)]['mean'])/self.bounds[(s, idx)]['bandwidth']
                                           for s, ts in enumerate(tick_sprds)])
                    if days_to_expiry_for_0 < 2:
                        argmin = np.argmin(normalized[1:]) + 1
                    else:
                        normalized[0] /= expiry_penalty_factor
                        argmin = np.argmin(normalized)

                    if argmin == idx:
                        continue

                    if normalized[argmin] < -self.boll_dev:
                        # print(bar_timestamp.isoformat(), normalized, argmin)
                        target_price = bars[vt_symbol].close_price - self.bounds[(idx, argmin)]['mean'] - \
                                       self.bounds[(argmin, idx)]['bandwidth'] * \
                                       self.boll_dev * (expiry_penalty_factor if argmin == 0 else 1)

                        self.buy(self.vt_symbols_today[argmin], target_price, current_pos_this_symbol)
                        close_price = bars[vt_symbol].close_price - self.price_add
                        self.switches.update({self.vt_symbols_today[argmin]: (vt_symbol, close_price)})
                        # self.sell(vt_symbol, close_price, current_pos_this_symbol)
                        # print(f"{bar_timestamp.isoformat()} - switch from idx {idx} "
                        #       f"({vt_symbol}) @{close_price:.2f} to idx"
                        #       f" {argmin} ({self.vt_symbols_today[argmin]} @{target_price:.2f})")
        else:
            pos_0 = self.get_pos(self.vt_symbols_today[0])
            if pos_0 > 0 and days_to_expiry_for_0 < 2:
                self.buy(self.vt_symbols_today[1], bars[self.vt_symbols_today[1]].close_price + self.price_add, pos_0)
                close_price = bars[self.vt_symbols_today[0]].close_price - self.price_add
                self.switches.update({self.vt_symbols_today[1]: (self.vt_symbols_today[0], close_price)})

        if sum(holdings) < self.target_position:
            print("buy future")
            self.buy(
                self.vt_symbols_today[1], bars[self.vt_symbols_today[1]].close_price + self.price_add,
                volume=self.target_position-sum(holdings)
            )

        short_spot_pos = self.get_pos(self.vt_symbol_spot)
        if short_spot_pos != -self.target_position:
            print(bar_timestamp.isoformat(), " - short spot")
            self.short(self.vt_symbol_spot, bars[self.vt_symbol_spot].close_price - self.price_add,
                       abs(-self.target_position - short_spot_pos))

        self.put_event()

    def update_order(self, order: OrderData) -> None:
        super(BackwardationRollingStrategy, self).update_order(order)
        # print(order)

    def update_trade(self, trade: TradeData) -> None:
        super(BackwardationRollingStrategy, self).update_trade(trade)
        if trade.vt_symbol in self.switches:
            vt_symbol, close_price = self.switches.pop(trade.vt_symbol)
            print(f"{trade.datetime.isoformat()} - switch from {vt_symbol} @{close_price:.2f} to"
                  f" {trade.vt_symbol} @{trade.price:.2f})")
            self.sell(vt_symbol, close_price, trade.volume)
