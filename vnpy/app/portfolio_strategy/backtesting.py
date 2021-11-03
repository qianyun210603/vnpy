from enum import Enum
from collections import defaultdict, OrderedDict, deque
from datetime import date, datetime, timedelta
from typing import Dict, List, Set, Optional, Union
from functools import lru_cache
from copy import copy
import traceback

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame

from vnpy.trader.constant import Direction, Offset, Interval, Status, Exchange
from vnpy.trader.database import get_database, DB_TZ
from vnpy.trader.object import OrderData, TradeData, BarData, TickData
from vnpy.trader.utility import round_to, extract_vt_symbol

from .template import StrategyTemplate



INTERVAL_DELTA_MAP = {
    Interval.TICK: timedelta(milliseconds=1),
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}


class BacktestingMode(Enum):
    BAR = 1
    TICK = 2


class BacktestingEngine:
    """"""

    gateway_name = "BACKTESTING"

    def __init__(self):
        """"""
        self.vt_symbols: List[str] = []
        self.start: Optional[datetime] = None
        self.end: Optional[datetime] = None

        self.intervals: Dict[str, Interval] = {}

        self.rates: Dict[str, float] = {}
        self.slippages: Dict[str, float] = {}
        self.sizes: Dict[str, float] = {}
        self.priceticks: Dict[str, float] = {}

        self.capital: float = 1_000_000
        self.risk_free: float = 0

        self.strategy: Optional[StrategyTemplate] = None
        self.bars: Dict[str, BarData] = {}
        self.datetime: Optional[datetime] = None

        self.interval: Optional[Interval] = None
        self.days: int = 0
        self.history_data: Dict[datetime, Dict[str, Union[TickData, BarData]]] = {}
        self.dts: Set[datetime] = set()

        self.limit_order_count = 0
        self.limit_orders = {}
        self.active_limit_orders = OrderedDict()
        self.limit_order_process_queue = deque()

        self.trade_count = 0
        self.trades = {}

        self.logs = []

        self.daily_results: Dict[date, PortfolioDailyResult] = {}
        self.daily_df = None

    def clear_data(self) -> None:
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.bars = {}
        self.datetime = None

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()
        self.daily_df = None

    def set_parameters(
        self,
        vt_symbols: List[str],
        interval: Interval,
        intervals: Dict[str, Interval],
        start: datetime,
        rates: Dict[str, float],
        slippages: Dict[str, float],
        sizes: Dict[str, float],
        priceticks: Dict[str, float],
        capital: int = 0,
        end: datetime = None,
        risk_free: float = 0
    ) -> None:
        """"""
        self.vt_symbols = vt_symbols
        self.interval = interval
        self.intervals = intervals

        self.rates = rates
        self.slippages = slippages
        self.sizes = sizes
        self.priceticks = priceticks

        self.start = start
        self.end = end
        self.capital = capital
        self.risk_free = risk_free

    def add_strategy(self, strategy_class: type, setting: dict) -> None:
        """"""
        self.strategy = strategy_class(
            self, strategy_class.__name__, copy(self.vt_symbols), setting
        )

    def load_data(self) -> None:
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        # Clear previously loaded history data
        self.history_data.clear()
        self.dts.clear()

        # Load 30 days of data each time and allow for progress update
        progress_delta = timedelta(days=300)
        total_delta = self.end - self.start

        for vt_symbol in self.vt_symbols:
            start = self.start
            end = self.start + progress_delta
            progress = 0
            symbol, exchange = extract_vt_symbol(vt_symbol)
            interval_delta = INTERVAL_DELTA_MAP[self.intervals.get(vt_symbol, self.interval)]

            data_count = 0
            real_start = datetime(2999, 1, 1, tzinfo=DB_TZ)
            real_end = datetime(1970, 1, 1, tzinfo=DB_TZ)
            while start < self.end:
                end = min(end, self.end)  # Make sure end time stays within set range

                if self.intervals.get(vt_symbol, self.interval) != Interval.TICK:
                    data = load_bar_data(
                        symbol,
                        exchange,
                        self.intervals.get(vt_symbol, self.interval),
                        start,
                        end
                    )
                    for d in data:
                        # self.dts.add(bar.datetime)
                        self.history_data.setdefault(
                            d.datetime + interval_delta - timedelta(microseconds=1), {}
                        )[vt_symbol] = d
                        data_count += 1
                        real_start = min(real_start, d.datetime)
                        real_end = max(real_end, d.datetime)
                else:
                    data = load_tick_data(
                        symbol,
                        exchange,
                        start,
                        end
                    )
                    for d in data:
                        # self.dts.add(bar.datetime)
                        # Since vnpy marks the timestamp at the beginning of the bar, for mixture backtesting, need
                        # to shift timestamp to end of bar to avoid forword looking
                        # (compare with tick or higher frequency bar)
                        self.history_data.setdefault(d.datetime, {})[vt_symbol] = d
                        data_count += 1
                        real_start = min(real_start, d.datetime)
                        real_end = max(real_end, d.datetime)

                # progress += progress_delta / total_delta
                # progress = min(progress, 1)
                # progress_bar = "#" * int(progress * 10)
                # self.output(f"{vt_symbol}加载进度：{progress_bar} [{progress:.0%}]")

                start = end + interval_delta
                end += (progress_delta + interval_delta)

            self.output(f"{vt_symbol}历史数据加载完成，"
                        f"从{real_start.isoformat()}到{real_end.isoformat()}，数据量：{data_count}")

        self.output("所有历史数据加载完成")

    def run_backtesting(self) -> None:
        """"""
        self.strategy.on_init()

        # Generate sorted datetime list
        dts = sorted(self.history_data)

        # Use the first [days] of history data for initializing strategy
        day_count = 0
        ix = 0

        for ix, dt in enumerate(dts):
            # if self.datetime and dt.day != self.datetime.day:
            #     print(self.datetime.isoformat(), dt.isoformat())
            #     day_count += 1
            #     if day_count >= self.days:
            #         break
            try:
                self.new_data(dt)
            except Exception:
                self.output("触发异常，回测终止")
                self.output(traceback.format_exc())
                return

            if ix != 0 and dts[ix-1].day != dts[ix].day:
                day_count += 1
                if day_count >= self.days:
                    break

        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        # Use the rest of history data for running backtesting
        for dt in dts[ix+1:]:
            try:
                self.new_data(dt)
            except Exception:
                self.output("触发异常，回测终止")
                self.output(traceback.format_exc())
                return

        self.output("历史数据回放结束")

    def calculate_result(self) -> DataFrame:
        """"""
        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("成交记录为空，无法计算")
            return DataFrame(columns=[
                "trade_count", "turnover",
                "commission", "slippage", "trading_pnl",
                "holding_pnl", "total_pnl", "net_pnl"
            ])

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)

        # Calculate daily result by iteration.
        pre_closes = {}
        start_poses = {}

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                self.sizes,
                self.rates,
                self.slippages,
            )

            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        # Generate dataframe
        results = defaultdict(list)

        for daily_result in self.daily_results.values():
            fields = [
                "date", "trade_count", "turnover",
                "commission", "slippage", "trading_pnl",
                "holding_pnl", "total_pnl", "net_pnl"
            ]
            for key in fields:
                value = getattr(daily_result, key)
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True):
        """"""
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date = ""
            end_date = ""
            total_days = 0
            profit_days = 0
            loss_days = 0
            end_balance = 0
            max_drawdown = 0
            max_ddpercent = 0
            max_drawdown_duration = 0
            total_net_pnl = 0
            daily_net_pnl = 0
            total_commission = 0
            daily_commission = 0
            total_slippage = 0
            daily_slippage = 0
            total_turnover = 0
            daily_turnover = 0
            total_trade_count = 0
            daily_trade_count = 0
            total_return = 0
            annual_return = 0
            daily_return = 0
            return_std = 0
            sharpe_ratio = 0
            return_drawdown_ratio = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * 240
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                daily_risk_free = self.risk_free / np.sqrt(240)
                sharpe_ratio = (daily_return - daily_risk_free) / return_std * np.sqrt(240)
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_net_pnl / max_drawdown

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("策略统计指标计算完成")
        return statistics

    def show_chart(self, df: DataFrame = None) -> None:
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def update_daily_close(self, dt: datetime, vt_symbol: str, price: float) -> None:
        """"""
        d = dt.date()

        # close_prices = {}
        # for bar in bars.values():
        #     close_prices[bar.vt_symbol] = bar.close_price

        if d not in self.daily_results:
            self.daily_results[d] = PortfolioDailyResult(d)

        self.daily_results[d].update_close_price(vt_symbol, price)

    def new_data(self, dt: datetime) -> None:
        """"""
        if self.datetime is None or self.datetime.day != dt.day:
            self.strategy.on_day_open(dt)
        self.datetime = dt

        for vt_symbol in self.history_data[dt]:
            self.strategy.update_latest_data(self.history_data[dt][vt_symbol])

        self.cross_limit_order(dt)
        self.limit_order_process_queue.extend(self.active_limit_orders)
        for vt_symbol, mkt_data in self.history_data[dt].items():
            if self.intervals.get(vt_symbol, self.interval) == Interval.TICK:
                self.strategy.on_tick(mkt_data)
                if self.strategy.inited:
                    self.update_daily_close(dt, vt_symbol, mkt_data.last_price)
            else:
                self.strategy.on_bar(mkt_data)
                if self.strategy.inited:
                    self.update_daily_close(dt, vt_symbol, mkt_data.close_price)

    def cross_limit_order(self, dt) -> None:
        """
        Cross limit order with last bar/tick data.
        """
        while bool(self.limit_order_process_queue):
            order: OrderData = self.active_limit_orders.get(self.limit_order_process_queue.popleft(), None)
            if order is None:
                continue
            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                order.datetime = dt
                self.strategy.update_order(order)

            if order.vt_symbol in self.history_data[dt]:
                mk_data = self.history_data[dt][order.vt_symbol]

                if self.intervals.get(order.vt_symbol, self.interval) != Interval.TICK:
                    long_cross_price = mk_data.low_price
                    short_cross_price = mk_data.high_price
                    long_best_price = mk_data.open_price
                    short_best_price = mk_data.open_price
                else:
                    last_price = (mk_data.bid_price_1 + mk_data.ask_price_1) / 2 if np.isnan(mk_data.last_price) \
                        else mk_data.last_price
                    long_cross_price = mk_data.ask_price_1
                    short_cross_price = mk_data.bid_price_1
                    long_best_price = max(mk_data.ask_price_1, last_price)
                    short_best_price = min(mk_data.bid_price_1, last_price)

                # Check whether limit orders can be filled.
                long_cross = (
                        order.direction == Direction.LONG
                        and order.price >= long_cross_price > 0
                )

                short_cross = (
                    order.direction == Direction.SHORT
                    and order.price <= short_cross_price
                    and short_cross_price > 0
                )

                if not long_cross and not short_cross:
                    continue

                # Push order update with status "all traded" (filled).
                order.traded = order.volume
                order.status = Status.ALLTRADED
                order.datetime = dt
                self.strategy.update_order(order)

                self.active_limit_orders.pop(order.vt_orderid)

                # Push trade update
                self.trade_count += 1

                if long_cross:
                    trade_price = min(order.price, long_best_price)
                else:
                    trade_price = max(order.price, short_best_price)

                trade = TradeData(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    orderid=order.orderid,
                    tradeid=str(self.trade_count),
                    direction=order.direction,
                    offset=order.offset,
                    price=trade_price,
                    volume=order.volume,
                    datetime=self.datetime,
                    gateway_name=self.gateway_name,
                )

                self.strategy.update_trade(trade)
                self.trades[trade.vt_tradeid] = trade

    def load_bars(
        self,
        _: StrategyTemplate,
        days: int,
        __: Interval
    ) -> None:
        """"""
        self.days = days

    def send_order(
        self,
        _: StrategyTemplate,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        __: bool,
        ___: bool
    ) -> List[str]:
        """"""
        price = round_to(price, self.priceticks[vt_symbol])
        symbol, exchange = extract_vt_symbol(vt_symbol)

        self.limit_order_count += 1

        order = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        self.active_limit_orders[order.vt_orderid] = order
        if self.intervals.get(order.vt_symbol, self.interval) != Interval.TICK:
            self.limit_order_process_queue.append(order.vt_orderid)
        self.limit_orders[order.vt_orderid] = order
        self.strategy.update_order(order)

        return [order.vt_orderid]

    def cancel_order(self, _: StrategyTemplate, vt_orderid: str) -> None:
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid not in self.active_limit_orders:
            return
        order = self.active_limit_orders.pop(vt_orderid)
        if vt_orderid in self.limit_order_process_queue:
            self.limit_order_process_queue.remove(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, _: StrategyTemplate = None) -> None:
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """
        Send email to default receiver.
        """
        pass

    def sync_strategy_data(self, strategy: StrategyTemplate) -> None:
        """
        Sync strategy data into json file.
        """
        pass

    def get_pricetick(self, _: StrategyTemplate, vt_symbol) -> float:
        """
        Return contract pricetick data.
        """
        return self.priceticks[vt_symbol]

    def put_strategy_event(self, strategy: StrategyTemplate) -> None:
        """
        Put an event to update strategy status.
        """
        pass

    def output(self, msg) -> None:
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self) -> List[TradeData]:
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self) -> List[OrderData]:
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> List["PortfolioDailyResult"]:
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())


class ContractDailyResult:
    """"""

    def __init__(self, result_date: date, close_price: float):
        """"""
        self.date: date = result_date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: List[TradeData] = []
        self.trade_count: int = 0

        self.start_pos: float = 0
        self.end_pos: float = 0

        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: float,
        rate: float,
        slippage: float
    ) -> None:
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            turnover = trade.volume * size * trade.price

            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.slippage += trade.volume * size * slippage
            self.turnover += turnover
            self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage

    def update_close_price(self, close_price: float) -> None:
        """"""
        self.close_price = close_price


class PortfolioDailyResult:
    """"""

    def __init__(self, result_date: date):
        """"""
        self.date: date = result_date
        self.close_prices: Dict[str, float] = {}
        self.pre_closes: Dict[str, float] = {}
        self.start_poses: Dict[str, float] = {}
        self.end_poses: Dict[str, float] = {}

        self.contract_results: Dict[str, ContractDailyResult] = {}

        self.trade_count: int = 0
        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0
        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """"""
        contract_result = self.contract_results[trade.vt_symbol]
        contract_result.add_trade(trade)

    def calculate_pnl(
        self,
        pre_closes: Dict[str, float],
        start_poses: Dict[str, float],
        sizes: Dict[str, float],
        rates: Dict[str, float],
        slippages: Dict[str, float],
    ) -> None:
        """"""
        self.pre_closes = pre_closes

        for vt_symbol, contract_result in self.contract_results.items():
            contract_result.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes[vt_symbol],
                rates[vt_symbol],
                slippages[vt_symbol]
            )

            self.trade_count += contract_result.trade_count
            self.turnover += contract_result.turnover
            self.commission += contract_result.commission
            self.slippage += contract_result.slippage
            self.trading_pnl += contract_result.trading_pnl
            self.holding_pnl += contract_result.holding_pnl
            self.total_pnl += contract_result.total_pnl
            self.net_pnl += contract_result.net_pnl

            self.end_poses[vt_symbol] = contract_result.end_pos

    def update_close_price(self, vt_symbol: str, price: float) -> None:
        """"""
        self.close_prices[vt_symbol] = price

        self.contract_results.setdefault(vt_symbol, ContractDailyResult(self.date, price)).update_close_price(price)


@lru_cache(maxsize=999)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime
):
    """"""
    database = get_database()

    return database.load_bar_data(
        symbol, exchange, interval, start, end
    )


@lru_cache(maxsize=999)
def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime
):
    """"""
    database = get_database()

    return database.load_tick_data(
        symbol, exchange, start, end
    )
