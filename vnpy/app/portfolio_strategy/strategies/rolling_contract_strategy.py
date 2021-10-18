import os
import os.path
import pickle
import pandas as pd
from jqdatasdk import auth, is_auth, get_all_securities
from typing import List, Dict, Optional
from datetime import datetime, time
from dateutil import tz

import numpy as np

from vnpy.app.portfolio_strategy import StrategyTemplate, StrategyEngine
from vnpy.trader.utility import BarGenerator
from vnpy.trader.object import TickData, BarData, TradeData, OrderData
# from vnpy.trader.constant import Status


class BackwardationRollingStrategy(StrategyTemplate):
    """"""

    author = "Booksword"
    price_add = 15
    band_floor = 3
    band_ceil = 100
    boll_window = 1200
    boll_multi_m = 1
    boll_multi_fm = 100
    boll_multi_q = 100

    target_position = 0

    current_spread = 0.0
    boll_mid = 0.0
    boll_down = 0.0
    boll_up = 0.0

    parameters = [
        "band_floor",
        "band_ceil",
        "boll_window",
        "boll_multi_m",
        "boll_multi_fm",
        "boll_multi_q",
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

        self.symbol_mapping: Dict[str, str] = {}
        self.contract_info = None
        self.vt_symbol_spot = vt_symbols[0]
        self.debug_file = None
        self._load_auxiliary_data()
        self.vt_symbols_today: List[str] = []
        self.expiries: List[datetime] = []
        self.days_to_expiry: List[int] = []
        self.expiries_ratio_to_main: List[float] = []
        self.latest_spot = 0

        self.switches = {}
        self.switch_mapping = {}

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
        self.debug_file = open(os.path.join(cache_path, "debug.txt"), "w")

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bars(self.boll_window // 240 + 1)

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
        self.days_to_expiry = [(e - today).days for e in self.expiries]
        self.expiries_ratio_to_main = [self.days_to_expiry[1] / max(1, nd) for nd in self.days_to_expiry]
        # print(self.vt_symbols_today)
        # print([(x - today.replace(hour=0, minute=0, second=0)).days for x in self.expiries])
        # for i in range(self.contracts_same_day):
        #     for j in range(self.contracts_same_day):
        #         std = self.spread_datas[(i, j)].std()
        #         mean = self.spread_datas[(i, j)].mean()
        #         params = {
        #             'mean': mean,
        #             'std': std,
        #             'bandwidth': max(std, self.band_floor)
        #         }
        #         self.bounds[(i, j)] = params
                # print(today, (i, j), str(params))
        # print(today, self.bounds[(1, 2)])
        # print([self.get_pos(vt_s) for vt_s in self.vt_symbols_today])

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

        trading = self.trading and time(9, 31) < self.last_tick_time.time() < time(14, 55)

        if not trading or not all(vt_sym in self.ticks for vt_sym in self.vt_symbols_today) or bool(self.switches):
            return

        # total = sum(holdings)
        # print(f"{bar_timestamp} - holdings: {str(holdings)}, total: {total}")
        # if total > self.target_position:
        # print(f"{bar_timestamp} - before order holdings: {str(holdings)}, total: {total}")
        ticks: List[TickData] = [self.ticks[vt_s] for vt_s in self.vt_symbols_today]

        backwardations = [min(-5.0, t.ask_price_1 - self.latest_spot) for t in ticks]
        multipliers = np.array([
            self.boll_multi_fm * np.exp(0.3 * (5 - min(5, self.days_to_expiry[0] - 2))), # for current month contract, adjust by expiry
            1.0, # no adjustment for main contract
            self.boll_multi_q * backwardations[1] / backwardations[2] / self.expiries_ratio_to_main[2],
            self.boll_multi_q * backwardations[1] / backwardations[3] / self.expiries_ratio_to_main[3],
        ])
        sprd_2_main_bid = np.array([t.bid_price_1 - ticks[1].ask_price_1 for t in ticks])
        sprd_2_main_ask = np.array([t.ask_price_1 - ticks[1].bid_price_1 for t in ticks])
        normalized_bid = (sprd_2_main_bid - self.means[:, 1]) / (self.bands[:, 1] * self.boll_multi_m)
        normalized_ask = (sprd_2_main_ask - self.means[:, 1]) / (self.bands[:, 1] * multipliers)
        if self.days_to_expiry[0] < 2:
            normalized_bid[0] = 100000
            normalized_ask[0] = 100000

        to_main = np.argwhere(normalized_bid > 1.0).flatten()
        from_main_argmin = normalized_ask.argmin()
        argmin = from_main_argmin if normalized_ask[from_main_argmin] < -1.0 else 1
        idx_to_move = to_main if argmin == 1 else np.append(to_main, 1)
        holdings = [self.get_pos(vt_s) for vt_s in self.vt_symbols_today]
        self.debug_file.write(tick.datetime.isoformat() + "\n")
        self.debug_file.write(f"backwardations: {str(backwardations)}\n")
        for t in ticks:
            self.debug_file.write(f"{t.vt_symbol}: {t.bid_price_1}, {t.ask_price_1}  ")
        self.debug_file.write('\n')
        self.debug_file.write(f"days_to_expiry: {str(self.days_to_expiry)}\n")
        self.debug_file.write(f"multipliers: {str(multipliers)}\n")
        self.debug_file.write(f"sprd_2_main_bid: {str(sprd_2_main_bid)}\n")
        self.debug_file.write(f"sprd_2_main_ask: {str(sprd_2_main_ask)}\n")
        self.debug_file.write(f"mean: {str(self.means[:, 1])}\n")
        self.debug_file.write(f"std: {str(self.stds[:, 1])}\n")
        self.debug_file.write(f"band: {str(self.bands[:, 1])}\n")
        self.debug_file.write(f"normalized_bid: {str(normalized_bid)}\n")
        self.debug_file.write(f"normalized_ask: {str(normalized_ask)}\n")

        for idx in idx_to_move:
            if holdings[idx] > 0:
                if self.days_to_expiry[idx] < 2:
                    target_price = ticks[argmin].ask_price_1 + self.price_add
                elif argmin == 1:
                    target_price = ticks[idx].bid_price_1 - self.means[idx, 1] - self.bands[idx, 1] * self.boll_multi_m
                else:
                    target_price = ticks[1].ask_price_1 + self.means[argmin, 1] + self.bands[argmin, 1] * multipliers[argmin]
                # target_price = np.floor(target_price / 0.2) * 0.2
                self.buy(self.vt_symbols_today[argmin], target_price, holdings[idx])
                close_price = ticks[idx].bid_price_1 - self.price_add

                self.switches.update({
                    self.vt_symbols_today[argmin]: [self.vt_symbols_today[idx], close_price, holdings[idx], 0]
                })
                self.switch_mapping[self.vt_symbols_today[idx]] = self.vt_symbols_today[argmin]
                self.debug_file.write(f"{self.vt_symbols_today[idx]}@{close_price}->{self.vt_symbols_today[argmin]}@{target_price}\n")

        # for idx, vt_symbol in enumerate(self.vt_symbols_today):
        #     current_pos_this_symbol = self.get_pos(vt_symbol)
        #     if current_pos_this_symbol > 0:
        #         tick_sprds = np.array([ticks[s].ask_price_1 - ticks[vt_symbol].bid_price_1
        #                                for s in self.vt_symbols_today])
        #         means = self.means[:, idx]
        #         stds = self.stds[:, idx]
        #         bws = self.bands[:, idx]
        #         print(f"tick spreads to {vt_symbol}: {str(tick_sprds)}, means: {str(means)}, stds: {str(stds)}, "
        #               f"bandwidths: {str(bws)}")
        #         normalized = np.array([(ts - self.means[s, idx]) / self.bands[s, idx] for s, ts in enumerate(tick_sprds)])
        #         if self.days_to_expiry[0] < 2:
        #             argmin = np.argmin(normalized[1:]) + 1
        #         else:
        #             normalized[0] /= expiry_penalty_factor
        #             argmin = np.argmin(normalized)
        #
        #         if argmin == idx:
        #             return
        #         if idx == 0 and days_to_expiry_for_0 < 2:
        #             print(tick.datetime.isoformat(), f'switch {vt_symbol} to {self.vt_symbols_today[argmin]}')
        #             target_price = ticks[self.vt_symbols_today[argmin]].ask_price_1 + self.price_add
        #             self.buy(self.vt_symbols_today[argmin], target_price, current_pos_this_symbol)
        #             close_price = ticks[vt_symbol].bid_price_1 - self.price_add
        #             self.switches.update({
        #                 self.vt_symbols_today[argmin]: [vt_symbol, close_price, current_pos_this_symbol, 0]
        #             })
        #             self.switch_mapping[vt_symbol] = self.vt_symbols_today[argmin]
        #         elif normalized[argmin] < self.boll_dev:
        #             # print(bar_timestamp.isoformat(), normalized, argmin)
        #             target_price = ticks[vt_symbol].bid_price_1 - self.means[idx, argmin] - \
        #                            self.bands[argmin, idx] * self.boll_dev * \
        #                            (expiry_penalty_factor if argmin == 0 else 1.0)
        #
        #             self.buy(self.vt_symbols_today[argmin], target_price, current_pos_this_symbol)
        #             close_price = ticks[vt_symbol].bid_price_1 - self.price_add
        #             self.switches.update({
        #                 self.vt_symbols_today[argmin]: [vt_symbol, close_price, current_pos_this_symbol, 0]
        #             })
        #             self.switch_mapping[vt_symbol] = self.vt_symbols_today[argmin]
        #             # self.sell(vt_symbol, close_price, current_pos_this_symbol)
        #             print(f"{tick.datetime.isoformat()} - switch from idx {idx} "
        #                   f"({vt_symbol}) @ {close_price:.2f} to idx"
        #                   f" {argmin} ({self.vt_symbols_today[argmin]} @{target_price:.2f})")

    def on_bar(self, bar: BarData) -> None:
        self.minute_bars[bar.vt_symbol] = bar
        if bar.vt_symbol == self.vt_symbol_spot:
            self.latest_spot = bar.close_price
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
        bars: List[BarData] = [self.minute_bars[vt_s] for vt_s in self.vt_symbols_today]
        bar_timestamp = bars[0].datetime
        # print(bar_timestamp.isoformat())
        for i in range(self.contracts_same_day):
            for j in range(self.contracts_same_day):
                current_spread = bars[i].close_price - bars[j].close_price
                self.spread_datas[i, j, :-1] = self.spread_datas[i, j, 1:]
                self.spread_datas[i, j, -1] = current_spread
                self.means[i, j] = self.spread_datas[i, j].mean()
                self.stds[i, j] = self.spread_datas[i, j].std()
                self.bands[i, j] = np.clip(self.stds[i, j], self.band_floor, self.band_ceil)
        if not self.trading:
            return

        holdings = [self.get_pos(vt_s) for vt_s in self.vt_symbols_today]

        if sum(holdings) < self.target_position:
            print(bar_timestamp.isoformat(), " - buy future")
            self.buy(
                self.vt_symbols_today[1], bars[1].close_price + self.price_add,
                volume=self.target_position-sum(holdings)
            )

        short_spot_pos = self.get_pos(self.vt_symbol_spot)
        if short_spot_pos != -self.target_position:
            print(bar_timestamp.isoformat(), " - short spot")
            self.short(self.vt_symbol_spot, self.minute_bars[self.vt_symbol_spot].close_price - self.price_add,
                       abs(-self.target_position - short_spot_pos))


        if bool(self.ticks) or not bool(self.minute_bars) or bool(self.switches):
            if not bool(self.ticks):
                print(bar_timestamp.isoformat(), 'here')
            self.put_event()
            return
        backwardations = [min(-5.0, t.close_price - self.latest_spot) for t in bars]
        multipliers = np.array([
            self.boll_multi_fm * np.exp(0.3 * (5 - min(5, self.days_to_expiry[0] - 2))),
            # for current month contract, adjust by expiry
            1.0,  # no adjustment for main contract
            self.boll_multi_q * backwardations[1] / backwardations[2] / self.expiries_ratio_to_main[2],
            self.boll_multi_q * backwardations[1] / backwardations[3] / self.expiries_ratio_to_main[3],
        ])
        sprd_2_main_bid = np.array([t.close_price - bars[1].close_price for t in bars])
        sprd_2_main_ask = np.array([t.close_price - bars[1].close_price for t in bars])
        normalized_bid = (sprd_2_main_bid - self.means[:, 1]) / (self.bands[:, 1] * self.boll_multi_m)
        normalized_ask = (sprd_2_main_ask - self.means[:, 1]) / (self.bands[:, 1] * multipliers)
        if self.days_to_expiry[0] < 2:
            normalized_bid[0] = 100
            normalized_ask[0] = 100
        to_main = np.argwhere(normalized_bid > 1.0).flatten()
        from_main_argmin = normalized_ask.argmin()
        self.debug_file.write(bar_timestamp.isoformat() + "\n")
        self.debug_file.write(f"backwardations: {str(backwardations)}\n")
        for t in bars:
            self.debug_file.write(f"{t.vt_symbol}: {t.close_price}, {t.close_price}  ")
        self.debug_file.write('\n')
        self.debug_file.write(f"days_to_expiry: {str(self.days_to_expiry)}\n")
        self.debug_file.write(f"multipliers: {str(multipliers)}\n")
        self.debug_file.write(f"sprd_2_main_bid: {str(sprd_2_main_bid)}\n")
        self.debug_file.write(f"sprd_2_main_ask: {str(sprd_2_main_ask)}\n")
        self.debug_file.write(f"mean: {str(self.means[:, 1])}\n")
        self.debug_file.write(f"std: {str(self.stds[:, 1])}\n")
        self.debug_file.write(f"band: {str(self.bands[:, 1])}\n")
        self.debug_file.write(f"normalized_bid: {str(normalized_bid)}\n")
        self.debug_file.write(f"normalized_ask: {str(normalized_ask)}\n")
        argmin = from_main_argmin if normalized_ask[from_main_argmin] < -1.0 else 1
        idx_to_move = to_main if argmin == 1 else np.append(to_main, 1)
        for idx in idx_to_move:
            if holdings[idx] > 0:
                if self.days_to_expiry[idx] < 2:
                    target_price = bars[argmin].close_price + self.price_add
                elif argmin == 1:
                    target_price = bars[idx].close_price - self.means[idx, 1] - self.bands[idx, 1] * self.boll_multi_m
                else:
                    target_price = bars[1].close_price + self.means[argmin, 1] + self.bands[argmin, 1] * multipliers[
                        argmin]
                target_price = np.floor(target_price/0.2) * 0.2
                self.buy(self.vt_symbols_today[argmin], target_price, holdings[idx])
                close_price = bars[idx].close_price - self.price_add
                self.switches.update({
                    self.vt_symbols_today[argmin]: [self.vt_symbols_today[idx], close_price, holdings[idx], 0]
                })
                self.switch_mapping[self.vt_symbols_today[idx]] = self.vt_symbols_today[argmin]
                self.debug_file.write(
                    f"{self.vt_symbols_today[idx]}@{close_price}->{self.vt_symbols_today[argmin]}@{target_price}\n")
        self.put_event()
        # days_to_expiry_for_0 = (self.expiries[0] - bar_timestamp).days
        #
        #
        # expiry_penalty_factor = np.exp(0.3 * (5 - min(5, days_to_expiry_for_0 - 2)))
        # # total = sum(holdings)
        # # print(f"{bar_timestamp} - holdings: {str(holdings)}, total: {total}")
        # # if total > self.target_position:
        # # print(f"{bar_timestamp} - before order holdings: {str(holdings)}, total: {total}")
        # for idx, vt_symbol in enumerate(self.vt_symbols_today):
        #     current_pos_this_symbol = self.get_pos(vt_symbol)
        #     if current_pos_this_symbol > 0:
        #         tick_sprds = [bars[s].close_price - bars[vt_symbol].close_price for s in self.vt_symbols_today]
        #         normalized = np.array([(ts-self.means[(s, idx)])/self.bands[s, idx]
        #                                for s, ts in enumerate(tick_sprds)])
        #         if days_to_expiry_for_0 < 2:
        #             argmin = np.argmin(normalized[1:]) + 1
        #         else:
        #             normalized[0] /= expiry_penalty_factor
        #             argmin = np.argmin(normalized)
        #
        #         if argmin == idx:
        #             continue
        #
        #         if idx == 0 and days_to_expiry_for_0 < 2:
        #             target_price = bars[self.vt_symbols_today[argmin]].close_price + self.price_add
        #             self.buy(self.vt_symbols_today[argmin], target_price, current_pos_this_symbol)
        #             close_price = bars[vt_symbol].close_price - self.price_add
        #             self.switches.update({
        #                 self.vt_symbols_today[argmin]: [vt_symbol, close_price, current_pos_this_symbol, 0]
        #             })
        #             self.switch_mapping[vt_symbol] = self.vt_symbols_today[argmin]
        #         elif normalized[argmin] < -self.boll_dev:
        #             # print(bar_timestamp.isoformat(), normalized, argmin)
        #             target_price = bars[vt_symbol].close_price - self.means[idx, argmin] - \
        #                            self.bands[argmin, idx] * self.boll_dev * \
        #                            (expiry_penalty_factor if argmin == 0 else 1)
        #
        #             self.buy(self.vt_symbols_today[argmin], target_price, current_pos_this_symbol)
        #             close_price = bars[vt_symbol].close_price - self.price_add
        #             self.switches.update({
        #                 self.vt_symbols_today[argmin]: [vt_symbol, close_price, current_pos_this_symbol, 0]
        #             })
        #             self.switch_mapping[vt_symbol] = self.vt_symbols_today[argmin]

    def update_order(self, order: OrderData) -> None:
        super(BackwardationRollingStrategy, self).update_order(order)
        # if order.datetime.replace(tzinfo=None) > datetime(2021, 3, 5):
        #     pass
        #     print(order)

    def update_trade(self, trade: TradeData) -> None:
        super(BackwardationRollingStrategy, self).update_trade(trade)
        print(trade)
        if trade.vt_symbol in self.switches:
            from_vt_symbol, close_price = self.switches[trade.vt_symbol][:2]
            # print(f"{trade.datetime.isoformat()} - switch from {from_vt_symbol} @{close_price:.2f} to"
            #       f" {trade.vt_symbol} @{trade.price:.2f})")
            self.switches[trade.vt_symbol][2] -= trade.volume
            self.sell(from_vt_symbol, close_price, trade.volume)
            self.switches[trade.vt_symbol][3] += trade.volume
            self.debug_file.write(
                f"{from_vt_symbol}->{trade.vt_symbol}@{trade.price}\n")

        if trade.vt_symbol in self.switch_mapping:
            to_vt_symbol = self.switch_mapping[trade.vt_symbol]
            self.switches[to_vt_symbol][3] -= trade.volume
            if self.switches[to_vt_symbol][2] == 0 and self.switches[to_vt_symbol][3] == 0:
                del self.switches[to_vt_symbol]
                del self.switch_mapping[trade.vt_symbol]
            self.debug_file.write(
                f"{trade.vt_symbol}@{trade.price}->{to_vt_symbol}\n")

