import os
import os.path
import csv
import argparse
from datetime import datetime, time
from vnpy.trader.constant import Exchange
from vnpy.trader.database import get_database
from vnpy.trader.object import TickData


def run_load_csv(root_path):
    """
    遍历同一文件夹内所有csv文件，并且载入到数据库中
    """
    database_manager = get_database()
    for file in os.listdir(root_path):
        if not file.endswith(".csv"):
            continue

        csv_load(database_manager, os.path.join(root_path, file), exchange=Exchange.CFFEX)
        # os.remove(os.path.join(root_path, file))


def csv_load(database_manager, file, exchange):
    """
    读取csv文件内容，并写入到数据库中
    """
    base_name = os.path.basename(file)
    symbol: str = base_name.split('_')[0]
    if symbol.endswith('次主力连续') or '月' in symbol or '季' in symbol:
        return
    if symbol.endswith('主力连续'):
        symbol = symbol.replace('主力连续', "88")

    print("载入文件：", file)
    with open(file, "r") as f:
        reader = csv.DictReader(f)

        ticks = []
        start = None
        count = 0

        for item in reader:

            # generate datetime
            date = item["交易日"]
            second = item["最后修改时间"]
            millisecond = item["最后修改毫秒"]

            standard_time = date + " " + second + "." + millisecond
            dt = datetime.strptime(standard_time, "%Y%m%d %H:%M:%S.%f")

            # filter
            if dt.time()<time(9, 30) or dt.time()>time(15, 0) or time(11, 30)<dt.time()<time(13, 0):
                continue

            tick = TickData(
                symbol=symbol,
                datetime=dt,
                exchange=exchange,
                last_price=float(item["最新价"]),
                last_volume=float(item["数量"]),
                volume=float(item["持仓量"]),
                turnover=float(item["成交金额"]),
                limit_up=float(item["涨停板价"]),
                limit_down=float(item["跌停板价"]),
                open_price=float(item["今开盘"]),
                high_price=float(item["最高价"]),
                low_price=float(item["最低价"]),
                pre_close=float(item["昨收盘"]),
                bid_price_1=float(item["申买价一"]),
                bid_volume_1=float(item["申买量一"]),
                ask_price_1=float(item["申卖价一"]),
                ask_volume_1=float(item["申卖量一"]),
                gateway_name="DB",
                localtime=dt
            )
            ticks.append(tick)

            # do some statistics
            count += 1
            if not start:
                start = tick.datetime

        if ticks:
            end = ticks[-1].datetime
            database_manager.save_tick_data(ticks)

            print("插入数据", start, "-", end, "总数量：", count)
        else:
            print("无数据可供插入")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save tick data from jinshuyuan to vnpy.')
    parser.add_argument('-path',
                        default=r'G:\FutSF_TickKZ_CTP_Daily_2020\if2102')  # source:path
    args = parser.parse_args()
    run_load_csv(args.path)
