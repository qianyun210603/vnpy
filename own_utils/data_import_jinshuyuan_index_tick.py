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

        csv_load(database_manager, os.path.join(root_path, file), exchange=Exchange.SSE)
        # os.remove(os.path.join(root_path, file))


def csv_load(database_manager, file, exchange):
    """
    读取csv文件内容，并写入到数据库中
    """
    base_name = os.path.basename(file)
    symbol: str = base_name.split('_')[0][2:]

    with open(file, "r") as f:
        reader = csv.DictReader(f)

        ticks = []
        start = None
        count = 0

        for item in reader:

            # generate datetime
            dt = datetime.strptime(item['时间'], "%Y-%m-%d %H:%M:%S")

            # filter
            if dt.time() > time(15, 0) or time(11, 30) < dt.time() < time(13, 0):
                continue
            if dt.time() < time(9, 30):
                dt = dt.replace(hour=9, minute=30, second=0)

            tick = TickData(
                symbol=symbol,
                datetime=dt,
                exchange=exchange,
                last_price=float(item["最新价"]),
                volume=float(item["成交量"]),
                turnover=float(item["成交额"]),
                limit_up=float('inf'),
                limit_down=0.0,
                bid_price_1=float(item["最新价"]),
                ask_price_1=float(item["最新价"]),
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

            print("处理文件：", base_name, " 插入数据", start, "-", end, "总数量：", count)
        else:
            print("处理文件：", base_name, "无数据可供插入")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save tick data from jinshuyuan to vnpy.')
    parser.add_argument('-path',
                        default=r'F:\BaiduNetdiskDownload\ToCompress')  # source:path
    args = parser.parse_args()
    run_load_csv(args.path)
