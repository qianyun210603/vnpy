# flake8: noqa
import importlib
from vnpy.event import EventEngine

from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp, GatewayAppSelectPanel
from functools import partial

gw_candidates = [
    "Ctp",
    "Ctptest",
    "Mini",
    "Femas",
    "Sopt",
    "Sec",
    "Uft",
    "Esunny",
    "Xtp",
    "ToraStock",
    "ToraOption",
    "Comstar",
    "Ib",
    "Tap",
    "Da",
    "Rohon",
    "Tts",
    "Ost",
    "Gtja",
]
app_candidates = [
    "PaperAccount",
    "CtaStrategy",
    "CtaBacktester",
    "SpreadTrading",
    "AlgoTrading",
    "OptionMaster",
    "PortfolioStrategy",
    "ScriptTrader",
    "ChartWizard",
    "RpcService",
    "ExcelRtd",
    "DataManager",
    "DataRecorder",
    "RiskManager",
    "WebTrader",
    "PortfolioManager",
    "WZTDC",
]


def main():
    """"""
    qapp = create_qapp()

    event_engine = EventEngine()

    main_engine = MainEngine(event_engine)

    def add_components_to_main_engine(components, main_engine=None):
        for gw in components["gateways"]:
            gw_module = importlib.import_module(f"vnpy_{gw.lower()}")
            main_engine.add_gateway(getattr(gw_module, f"{gw}Gateway"))
        for app in components["apps"]:
            app_module = importlib.import_module(f"vnpy_{app.lower()}")
            main_engine.add_app(getattr(app_module, f"{app}App"))

    dialog = GatewayAppSelectPanel(available_gateways=gw_candidates, available_apps=app_candidates)
    dialog.checkbox_values_signal.connect(partial(add_components_to_main_engine, main_engine=main_engine))
    dialog.exec()

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()


if __name__ == "__main__":
    main()
