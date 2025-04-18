import json
from typing import Any, Dict, Tuple, Type, Literal, Optional

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation

from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np
from statistics import NormalDist

trades = 0

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Strategy(ABC):
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit

    def load_state(self, state: Any) -> None:
        return None 

    def save_state(self) -> Any:
        pass
 
    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders: list[Order] = []
        self.conversions = 0

        self.act(state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

class Macarons(Strategy):
    def __init__(self, symbol: str, position_limit: int, csi: float):
        super().__init__(symbol, position_limit)
        self.csi = csi
        self.state = 0
        self.previous_sunlightIndex = None
        self.time = 0
        self.sunlightIndex_grad = None

    def load_state(self, state: Tuple[int, Optional[float], int, Optional[float]]) -> None:
        self.state, self.previous_sunlightIndex, self.time, self.sunlightIndex_grad = state

    def save_state(self) -> Any:
        return self.state, self.previous_sunlightIndex, self.time, self.sunlightIndex_grad

    def update(self, obs: ConversionObservation) -> None:
        if self.previous_sunlightIndex is None:
            self.time += 1
            self.previous_sunlightIndex = obs.sunlightIndex
            return

        self.time += 1
        if round(obs.sunlightIndex,1 ) != round(self.previous_sunlightIndex, 1):
            self.sunlightIndex_grad = (obs.sunlightIndex - self.previous_sunlightIndex) / self.time
            self.time = 1 
            self.previous_sunlightIndex = obs.sunlightIndex
        logger.print(self.sunlightIndex_grad)

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.symbol, 0)
        order_depth: OrderDepth = state.order_depths[self.symbol]

        obs = state.observations.conversionObservations.get(self.symbol, None)
        self.update(obs)
        
        if self.state == 0:
            if obs.sunlightIndex < self.csi:
                self.state = 1
                
        if self.state == 1:
            if self.sunlightIndex_grad is not None:
                if self.sunlightIndex_grad > 0.003:
                    self.state = 0
        
        if self.state == 0:
            if position <= 0:
                self.convert(-1 * max(-10, position))
            
                best_ask = None
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())

                if obs is None:
                    return
                
                buy_price = obs.askPrice + obs.transportFees + obs.importTariff
                
                our_ask = max(int(buy_price + 1), best_ask - 1)
                
                effective_position_limit = 10
                self.sell(our_ask, effective_position_limit)
            else:
                if order_depth.buy_orders:
                    best_bid = min(order_depth.buy_orders.keys())
                    self.sell(best_bid, position)
        
        if self.state == 1:
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                price = best_ask
                self.buy(price, self.position_limit - position)

class Trader:
    def __init__(self) -> None:
        init_dict: dict[Symbol, Tuple[Type[Strategy], int, dict[str, Any]]] = {
                "MAGNIFICENT_MACARONS": (Macarons, 75, {"csi": 50})
            # Symbol: (Strategy, Position Limit, kwargs)
            # kwargs are for Strategy Parameters
                }
        self.strategies: dict[Symbol, Strategy] = {
            symbol: cls(symbol, position_limit, **kwargs) for symbol, (cls, position_limit, kwargs) in init_dict.items()
                }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        trader_data = ""
        
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load_state(old_trader_data[symbol])
             
            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                result[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save_state()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

