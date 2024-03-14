"""
Event-driven framework of VeighNa framework.
"""
import logging
from collections import defaultdict
from queue import Empty, Queue
from threading import Thread, get_ident
from time import sleep, time_ns
from typing import Any, Callable, List
from ..trader.object import TickData
from ..trader.utility import setup_plain_logger
from datetime import datetime

EVENT_TIMER = "eTimer"

logger = setup_plain_logger(__name__, logging.INFO, "event.log")

class Event:
    """
    Event object consists of a type string which is used
    by event engine for distributing event, and a data
    object which contains the real data.
    """

    def __init__(self, etype: str, data: Any = None) -> None:
        """"""
        self.type: str = etype
        self.data: Any = data


# Defines handler function to be used in event engine.
HandlerType: callable = Callable[[Event], None]


class EventEngine:
    """
    Event engine distributes event object based on its type
    to those handlers registered.

    It also generates timer event by every interval seconds,
    which can be used for timing purpose.
    """

    def __init__(self, interval: int = 1) -> None:
        """
        Timer event is generated every 1 second by default, if
        interval not specified.
        """
        self._interval: int = interval
        self._queue: Queue = Queue()
        self._active: bool = False
        self._thread: Thread = Thread(target=self._run)
        self._timer: Thread = Thread(target=self._run_timer)
        self._handlers: defaultdict = defaultdict(list)
        self._general_handlers: List = []

    def _run(self) -> None:
        """
        Get event from queue and then process it.
        """
        logger.info(f"Event engine started. Thread ID: {get_ident()}")
        while self._active:
            try:
                event: Event = self._queue.get(block=True, timeout=1)
                #ts = time_ns()
                self._process(event)
            except Empty:
                # logger.info(f"Event Queue empty. Thread ID: {get_ident()}")
                pass

    def _process(self, event: Event) -> None:
        """
        First distribute event to those handlers registered listening
        to this type.

        Then distribute event to those general handlers which listens
        to all types.
        """
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                handler(event)

        if self._general_handlers:
            for handler in self._general_handlers:
                handler(event)

    def _run_timer(self) -> None:
        """
        Sleep by interval second(s) and then generate a timer event.
        """
        while self._active:
            sleep(self._interval)
            event: Event = Event(EVENT_TIMER)
            self.put(event)

    def start(self) -> None:
        """
        Start event engine to process events and generate timer events.
        """
        self._active = True
        self._thread.start()
        self._timer.start()

    def stop(self) -> None:
        """
        Stop event engine.
        """
        self._active = False
        self._timer.join()
        self._thread.join()

    def put(self, event: Event) -> None:
        """
        Put an event object into event queue.
        """
        if event.type == 'eTick.':
            tick: TickData = event.data
            print(f"put in event queue [{tick.vt_symbol}] {datetime.now().isoformat()}, {tick.datetime.isoformat()}")
        self._queue.put(event)

    def register(self, etype: str, handler: HandlerType) -> None:
        """
        Register a new handler function for a specific event type. Every
        function can only be registered once for each event type.
        """
        handler_list: list = self._handlers[etype]
        if handler not in handler_list:
            handler_list.append(handler)

    def unregister(self, etype: str, handler: HandlerType) -> None:
        """
        Unregister an existing handler function from event engine.
        """
        handler_list: list = self._handlers[etype]

        if handler in handler_list:
            handler_list.remove(handler)

        if not handler_list:
            self._handlers.pop(type)

    def register_general(self, handler: HandlerType) -> None:
        """
        Register a new handler function for all event types. Every
        function can only be registered once for each event type.
        """
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def unregister_general(self, handler: HandlerType) -> None:
        """
        Unregister an existing general handler function.
        """
        if handler in self._general_handlers:
            self._general_handlers.remove(handler)
