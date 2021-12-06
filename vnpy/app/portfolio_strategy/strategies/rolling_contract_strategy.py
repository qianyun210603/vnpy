import os
import os.path
import pickle
import pandas as pd
from jqdatasdk import auth, is_auth, get_all_securities
from typing import List, Dict, Optional
from datetime import datetime, time
from dateutil import tz

import numpy as np
# import threading
from vnpy.app.portfolio_strategy import StrategyTemplate, StrategyEngine
from vnpy.trader.utility import BarGenerator
from vnpy.trader.object import TickData, BarData, TradeData, OrderData
# IF2112.CFFEX,IF2201.CFFEX,IF2203.CFFEX,IF2206.CFFEX


class BackwardationRollingStrategyM(StrategyTemplate):
    """"""

    author = "Booksword"
    price_add = 15
    band_floor = 5
    band_ceil = 100
    boll_window = 240
    boll_multi_m = 3
    boll_multi_fm = 100
    boll_multi_q = 100
    abandon_date = 0  # for backtesting only
    backtest = False

    target_position = 0
    start_contract_no = -1

    current_spread = 0.0
    boll_mid = 0.0
    boll_down = 0.0
    boll_up = 0.0
    boll_std = 0.0
    spread_0_1_bid = 0.0
    spread_0_1_ask = 0.0

    parameters = [
        "band_floor",
        "band_ceil",
        "boll_window",
        "boll_multi_m",
        "boll_multi_fm",
        "boll_multi_q",
        "target_position",
        "start_contract_no",
        "abandon_date",
        "backtest",
    ]
    variables = [
        "boll_mid",
        "boll_std",
        "boll_down",
        "boll_up",
        "spread_0_1_bid",
        "spread_0_1_ask",
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
        self.last_tick_time: Optional[datetime] = datetime(1970, 1, 1, tzinfo=tz.gettz('Asia/Shanghai'))
        self.last_bar_time: Optional[datetime] = datetime(1970, 1, 1, tzinfo=tz.gettz('Asia/Shanghai'))
        self.minute_bars: Dict[str, BarData] = {}
        self.underlying_symbol = vt_symbols[-1][:2]
        self.ticks: Dict[str, TickData] = {}

        self.spread_count: int = 0
        self.contracts_same_day = 4
        self.indexes: np.ndarray = np.arange(self.contracts_same_day)
        self.spread_datas: np.ndarray = np.zeros((self.contracts_same_day, self.contracts_same_day, self.boll_window))
        # Dict[Tuple[int, int], np.array] = {
        #     (i, j): np.zeros(self.boll_window) for i in range(self.contracts_same_day)
        #     for j in range(self.contracts_same_day)
        # }
        self.means: np.ndarray = np.zeros((self.contracts_same_day, self.contracts_same_day))
        self.stds: np.ndarray = np.ones((self.contracts_same_day, self.contracts_same_day))
        self.bands: np.ndarray = np.ones((self.contracts_same_day, self.contracts_same_day))
        # Dict[Tuple[int, int], Dict] = {}

        self.parameter_date: Optional[datetime] = None
        self.liquidity_adjust: np.ndarray = np.array([0.0, 0.0, 3, 10.0])

        self.symbol_mapping: Dict[str, str] = {}
        self.contract_info = None
        self.vt_symbol_spot = vt_symbols[0]
        self.debug_file = None
        self._load_auxiliary_data()
        self.vt_symbols_today: List[str] = []
        self.expiries: List[datetime] = []
        self.days_to_expiry: np.ndarray = np.array([])
        self.expiries_ratio_to_main: List[float] = []
        self.latest_spot = 0
        self.latest_spot_time = pd.Timestamp(year=1970, month=1, day=1)

        self.switches = {}
        self.switch_mapping = {}

        for vt_symbol in self.vt_symbols:
            self.targets[vt_symbol] = 0
            self.bgs[vt_symbol] = BarGenerator(self.on_bar)

    def __del__(self):
        self.write_log("strategy destroyed")
        # cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.__class__.__name__)
        # os.makedirs(cache_path, exist_ok=True)
        # status_path = os.path.join(cache_path, 'status.bin')
        # with open(status_path, "wb") as f:
        #     pickle.dump({'pos': self.pos}, f)

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
        self.contract_info = my_contracts.set_index('name').drop(['IF8888', 'IF9999'], errors='ignore')
        self.debug_file = open(os.path.join(cache_path, "debug.txt"), "w")

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.__class__.__name__)
        os.makedirs(cache_path, exist_ok=True)
        # status_path = os.path.join(cache_path, 'status.bin')
        # with open(status_path, "rb") as f:
        #     status_vars: dict = pickle.load(f)
        # for field, value in status_vars.items():
        #     setattr(self, field, value)
        self.load_bars(max(self.abandon_date, self.boll_window // 240 + 1))
        # self.write_log(str(self.pos))

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_day_open(self, today) -> None:
        today = pd.Timestamp(today, tz='Asia/Shanghai')
        raw = self.contract_info[
                (self.contract_info.start_date <= today) &
                (self.contract_info.end_date + pd.Timedelta(hours=23) >= today)
            ].sort_values(by='end_date').index.to_list()
        self.vt_symbols_today = [x + '.CFFEX' for x in raw]
        self.expiries = [x.to_pydatetime().replace(hour=16) for x in self.contract_info.loc[raw, 'end_date']]
        self.days_to_expiry = np.array([(e - today).days for e in self.expiries])

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        if self.debug_file is not None:
            self.debug_file.close()
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        if tick.datetime.date() > max(self.last_bar_time, self.last_tick_time).date():
            self.on_day_open(tick.datetime.date())
        self.ticks[tick.vt_symbol] = tick
        if (
            self.last_tick_time
            and self.last_tick_time.minute != tick.datetime.minute
        ):

            bg = self.bgs[tick.vt_symbol]
            # print(tick.vt_symbol, tick.datetime)
            bar: BarData = bg.generate()
            if bar:
                self.on_bar(bar)
            else:
                self.write_log("tick: %s" % str(tick))
        # from datetime import time
        # # if tick.datetime.time() > time(14, 59):
        # #     print(tick.datetime)
        bg: BarGenerator = self.bgs[tick.vt_symbol]
        bg.update_tick(tick)
        self.last_tick_time = tick.datetime

        if not self.trading:
            return

        if tick.datetime.time() < time(9, 33) or tick.datetime.time() > time(14, 57):
            return

        ticks: List[TickData] = [self.ticks.get(vt_s, None) for vt_s in self.vt_symbols_today[:2]]
        if any(t is None for t in ticks):
            return

        sources = set()
        target = 0

        holdings = [self.get_pos(vt_s) for vt_s in self.vt_symbols_today[:2]]

        if all(h == 0 for h in holdings) and not bool(self.get_all_active_orderids()):
            self.write_log("初始买入")
            self.buy(self.vt_symbols_today[0], ticks[0].ask_price_1+1, 1)
            if self.backtest:
                self.sell(self.vt_symbol_spot, self.ticks[self.vt_symbol_spot].bid_price_1-1, 1)
            return
        i, j = 0, 1
        self.spread_0_1_bid = ticks[i].bid_price_1 - ticks[j].ask_price_1
        self.spread_0_1_ask = ticks[i].ask_price_1 - ticks[j].bid_price_1
        if holdings[i] > 0 and (self.days_to_expiry[i] == 0 or self.spread_0_1_bid > self.boll_up):
            # print(self.days_to_expiry)
            sources.add(i)
            target = j
            self.write_log(
                str(holdings) + " " + f"({ticks[i].bid_price_1:.2f}, {ticks[i].ask_price_1:.2f}) --"
                                      f" ({ticks[j].bid_price_1:.2f}, {ticks[j].ask_price_1:.2f})"
            )
        elif holdings[j] > 0 and (self.days_to_expiry[i] > 0 and self.spread_0_1_ask < self.boll_down):
            # print(self.days_to_expiry)
            sources.add(j)
            target = i
            self.write_log(
                str(holdings) + " " + f"({ticks[i].bid_price_1:.2f}, {ticks[i].ask_price_1:.2f}) --"
                                      f" ({ticks[j].bid_price_1:.2f}, {ticks[j].ask_price_1:.2f})"
            )

        try:
            for f, t in list(self.switch_mapping.items()):
                fi = self.vt_symbols_today.index(f)
                ti = self.vt_symbols_today.index(t)
                if ti != target or fi not in sources:
                    for oid in self.switches[t][3]:
                        order = self.get_order(oid)
                        if order and order.is_active():
                            self.write_log(f"canceled: {oid}: {order.vt_symbol} {order.price} {order.direction.name}")
                            self.cancel_order(oid)
                    self.switches[t][3] = [oid for oid in self.switches[t][3]
                                           if oid in self.orders and self.orders[oid].is_active()]
                    if self.switches[t][1] >= self.switches[t][2]:
                        del self.switch_mapping[f]
                        del self.switches[t]
        except Exception:
            import traceback
            print(tick.datetime.isoformat())
            traceback.print_exc()
            raise

        if not bool(self.ticks) or bool(self.switches):
            return
        # print(holdings)
        for idx in sources:
            if holdings[idx] > 0 and idx != target:
                if self.days_to_expiry[idx] == 0:
                    target_price = ticks[target].ask_price_1 + self.price_add
                else:
                    target_price = max(ticks[target].ask_price_1 + 1, ticks[target].last_price-0.2)
                target_price = np.floor(target_price/0.2) * 0.2
                if self.vt_symbols_today[target] in self.switches:
                    active = 0
                    for oid in self.switches[self.vt_symbols_today[target]][3]:
                        o = self.get_order(oid)
                        if o and o.price <= target_price + 1e-4:
                            active += o.volume - o.traded
                        else:
                            print("replaced", oid, o.vt_symbol)
                            self.cancel_order(oid)
                            self.switches[self.vt_symbols_today[target]][3].remove(oid)
                    order_amount = holdings[idx] - active
                    # print(holdings[idx], active)
                    if order_amount > 0 and ticks[target].ask_price_1 <= target_price:
                        buy_id = self.buy(self.vt_symbols_today[target], target_price, order_amount)
                        self.switches[self.vt_symbols_today[target]][3].extend(buy_id)
                elif ticks[target].ask_price_1 <= target_price:
                    buy_id = self.buy(self.vt_symbols_today[target], target_price, holdings[idx])
                    self.switches.update({
                        self.vt_symbols_today[target]: [self.vt_symbols_today[idx], holdings[idx], 0, buy_id]
                    })
                    self.switch_mapping[self.vt_symbols_today[idx]] = self.vt_symbols_today[target]
                self.debug_file.write(
                    f"{self.vt_symbols_today[idx]}->{self.vt_symbols_today[target]}@{target_price}\n")

    def on_bar(self, bar: BarData) -> None:
        if bar.datetime.date() > max(self.last_bar_time, self.last_tick_time).date():
            self.on_day_open(bar.datetime.date())
        self.minute_bars[bar.vt_symbol] = bar
        if self.backtest and bar.vt_symbol == self.vt_symbol_spot:
            self.latest_spot = bar.close_price
        # print(bar.datetime.isoformat(), bar.vt_symbol, bar.open_price, bar.high_price, bar.low_price, bar.close_price)
        if bar.datetime > self.last_bar_time:
            self.last_bar_time = bar.datetime
        # print(self.last_bar_time.isoformat(), bar.vt_symbol, bar.datetime.isoformat(),
        # self.minute_bars[self.vt_symbol_spot].datetime)
        if all(s in self.minute_bars and self.last_bar_time == self.minute_bars[s].datetime
               for s in self.vt_symbols_today) and (
                not self.backtest or self.vt_symbol_spot in self.minute_bars and
                self.minute_bars[self.vt_symbol_spot].datetime == self.last_bar_time):
            self.on_bars()

    def on_bars(self):
        """"""
        bars: List[BarData] = [self.minute_bars[vt_s] for vt_s in self.vt_symbols_today]
        # bar_timestamp = bars[0].datetime
        # print(bar_timestamp.isoformat())
        for i in range(self.contracts_same_day):
            for j in range(self.contracts_same_day):
                current_spread = bars[i].close_price - bars[j].close_price
                self.spread_datas[i, j, :-1] = self.spread_datas[i, j, 1:]
                self.spread_datas[i, j, -1] = current_spread
                self.means[i, j] = self.spread_datas[i, j].mean()
                self.stds[i, j] = self.spread_datas[i, j].std()
                self.bands[i, j] = np.clip(self.stds[i, j]*self.boll_multi_m, self.band_floor, self.band_ceil)
        self.boll_mid = self.means[0, 1]
        self.boll_std = self.stds[0, 1]
        self.boll_up = self.means[0, 1] + self.bands[0, 1]
        self.boll_down = self.means[0, 1] - self.bands[0, 1]

        self.put_event()

    def update_order(self, order: OrderData) -> None:
        super(BackwardationRollingStrategyM, self).update_order(order)
        self.write_log(str(order))
        # if order.datetime.replace(tzinfo=None) > datetime(2021, 3, 5):
        #     pass
        #     print(order)

    def update_trade(self, trade: TradeData) -> None:
        super(BackwardationRollingStrategyM, self).update_trade(trade)
        # if bool(self.switches) or bool(self.switch_mapping):
        #     print("in trade", self.switches, self.switch_mapping)
        print(trade)
        if trade.vt_symbol in self.switches:
            from_vt_symbol = self.switches[trade.vt_symbol][0]
            # print(f"{trade.datetime.isoformat()} - switch from {from_vt_symbol} @{close_price:.2f} to"
            #       f" {trade.vt_symbol} @{trade.price:.2f})")
            self.switches[trade.vt_symbol][1] -= trade.volume
            close_price = self.ticks[from_vt_symbol].bid_price_1 - self.price_add if from_vt_symbol in self.ticks \
                else self.minute_bars[from_vt_symbol].open_price - self.price_add
            self.sell(from_vt_symbol, close_price, trade.volume)
            self.switches[trade.vt_symbol][2] += trade.volume
            self.debug_file.write(
                f"{from_vt_symbol}->{trade.vt_symbol}@{trade.price}\n")

        if trade.vt_symbol in self.switch_mapping:
            to_vt_symbol = self.switch_mapping[trade.vt_symbol]
            self.switches[to_vt_symbol][2] -= trade.volume
            if self.switches[to_vt_symbol][1] == 0 and self.switches[to_vt_symbol][2] == 0:
                del self.switches[to_vt_symbol]
                del self.switch_mapping[trade.vt_symbol]
            self.debug_file.write(
                f"{trade.vt_symbol}@{trade.price}->{to_vt_symbol}\n"
            )
