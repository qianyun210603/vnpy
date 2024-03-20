"""
Event type string used in the trading platform.
"""

from vnpy.event import EVENT_TIMER  # noqa, pylint: disable=unused-import

EVENT_TICK = "eTick."
EVENT_TRADE = "eTrade."
EVENT_ORDER = "eOrder."
EVENT_POSITION = "ePosition."
EVENT_ACCOUNT = "eAccount."
EVENT_QUOTE = "eQuote."
EVENT_CONTRACT = "eContract."
EVENT_LOG = "eLog"
EVENT_MARGINRATE = "eMarginRate"
EVENT_COMMISION = "eCommision"
