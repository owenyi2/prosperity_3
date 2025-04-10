import json
from typing import Any, Dict, Tuple, Type

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np

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

def market_make_take(strat: Strategy, position: int, position_limit: int, fair_value: int, order_depth: OrderDepth, spread: int) -> None:
    osell = OrderedDict(sorted(order_depth.sell_orders.items()))
    obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
     
    cpos = position
    for ask, vol in osell.items(): # vol < 0
        if ((ask < fair_value) or ((position < 0) and (ask == fair_value))):
            if cpos < position_limit:
                order_for = min(-vol, position_limit - cpos)
                cpos += order_for
                strat.buy(ask, order_for)

    our_bid = fair_value - spread 
    best_buy_pr = list(filter(lambda x: x < fair_value, obuy.keys()))
    if best_buy_pr:
        undercut_buy = best_buy_pr[0] + 1
        our_bid = min(undercut_buy, our_bid)
    if position == -position_limit:
        # when we hit a position_limit, we want to be more aggressive with our bids/asks in order to liquidate position faster
        our_bid = fair_value - spread // 2

    if cpos < position_limit:
        num = position_limit - cpos
        strat.buy(our_bid, num) 
        cpos += num

    cpos = position
    for bid, vol in obuy.items(): # vol > 0
        if ((bid > fair_value) or ((position > 0) and (bid == fair_value))):
            if cpos > -position_limit:
                order_for = min(vol, position_limit - cpos)
                cpos -= order_for
                strat.sell(bid, order_for)

    our_ask = fair_value + spread
    best_sell_pr = list(filter(lambda x: x > fair_value, osell.keys()))
    if best_sell_pr:
        undercut_sell = best_sell_pr[0] - 1
        our_ask = max(undercut_sell, our_ask)
    if position == position_limit:
        our_ask = fair_value + spread // 2

    if cpos > -position_limit:
        num = position_limit - (-cpos)
        strat.sell(our_ask, num)
        cpos -= num
     
class RainforestResin(Strategy):
    def __init__(self, symbol: str, position_limit: int, spread: int) -> None:
        super().__init__(symbol, position_limit)
        self.spread = spread

    def act(self, state: TradingState) -> None:
        position: int = state.position.get(self.symbol, 0)
        order_depth: OrderDepth = state.order_depths[self.symbol]
        fair_value = 10000
        
        market_make_take(self, position, self.position_limit, fair_value, order_depth, self.spread)
    
class SquidInk(Strategy):
    def __init__(self, symbol: str, position_limit: int, window_length: int, spread: int) -> None:
        super().__init__(symbol, position_limit)
        self.midprice: float = 2000
        self.window: list[float] = []

        self.window_length = window_length
        self.spread = 2

    def load_state(self, state: tuple[float, list[float]]) -> None:
        self.previous_midprice, self.window = state

    def save_state(self) -> tuple[float, list[float]]:
        return self.midprice, self.window 

    def update(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
         
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders:
            popular_buy_price = self.previous_midprice - 1
        else:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]

        if not sell_orders:
            popular_sell_price = self.previous_midprice + 1
        else:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        self.midprice = (popular_buy_price + popular_sell_price) / 2

        self.window.append(self.midprice)
        if len(self.window) > self.window_length:
            self.window = self.window[1:]
        
        weights = np.arange(1, self.window_length + 1, 1)[:len(self.window)]
        self.wma = np.dot(weights, self.window) / np.sum(weights)
        
    def market_make(self, position: int, our_bid: int, our_ask: int):
        cpos = position
        
        "Right so unlike kelp or resin, there aren't immediately obvious orders to take"
        if cpos < self.position_limit:
            num = self.position_limit - cpos
            self.buy(our_bid, num) 
            cpos += num

        cpos = position

        if cpos > -self.position_limit:
            num = self.position_limit - (-cpos)
            self.sell(our_ask, num)
            cpos -= num

    def quantile_forecast(self, value: float) -> float: # \in [0, 1]
        x_values = np.array([-10.        ,  -1.89090909,  -1.10909091,  -0.64545455,
            -0.28181818,   0.03636364,   0.35454545,   0.69090909,
             1.12727273,   1.78181818,  10.        ])
        y_values = np.linspace(-1, 1, 11)

        idx = np.searchsorted(x_values, value, side='right') - 1
        
        if idx < 0:
            return y_values[0]
        
        if idx >= len(x_values) - 1:
            return y_values[-1]
        
        x0 = x_values[idx]
        x1 = x_values[idx + 1]
        y0 = y_values[idx]
        y1 = y_values[idx + 1]
        interpolated_y = y0 + (value - x0) * (y1 - y0) / (x1 - x0)
    
        return interpolated_y
    
    def reach_position(self, position: int, desired_position: int, aggressiveness: int, order_depth: OrderDepth) -> None:
        change = abs(desired_position - position)
         
        osell = order_depth.sell_orders
        obuy = order_depth.buy_orders

        best_bid = max(obuy.keys())
        bid_vol = obuy[best_bid]
        best_ask = min(osell.keys())
        ask_vol = -osell[best_ask]

        midprice = int(round((best_bid + best_ask) / 2))

        if desired_position > position: # buy
            self.buy(best_bid, change) 

            # and then we want to make a market with volums around the desired position
        
        if desired_position < position: # sell
            self.sell(best_ask, change)

        # if desired_position > position: # buy
        #     match aggressiveness:
        #         case 0: # place a passive limit order
        #             self.buy(best_bid, change) 
        #         case 1: # place a limit order at the midprice
        #             self.buy(midprice, change)
        #         case 2: # place a market order and a limit order if necessary
        #             self.buy(best_ask, ask_vol)
        #             if ask_vol < change:
        #                 self.buy(midprice, change - ask_vol)
        #                  


        # if desired_position < position: # sell
        #     match aggressiveness:
        #         case 0:
        #             self.sell(best_ask, change)
        #         case 1:
        #             self.sell(midprice, change)
        #         case 2: 
        #             self.sell(best_bid, bid_vol)
        #             if bid_vol < change:
        #                 self.sell(midprice, change - ask_vol)
    
    def act(self, state: TradingState) -> None:
        position: int = state.position.get(self.symbol, 0)
        order_depth: OrderDepth = state.order_depths[self.symbol]
        self.update(state)
            
        forecast = self.quantile_forecast(self.midprice - self.wma)
        desired_position = int(round(self.position_limit * forecast))
        
        print(position, desired_position)
        # if abs(forecast) > 0.8:
        #     aggressiveness = 2
        # elif abs(forecast) > 0.4:
        #     aggressiveness = 1
        # else:
        #     aggressiveness = 0

        self.reach_position(position, desired_position, 0, order_depth)

class Trader:
    def __init__(self) -> None:
        init_dict: dict[Symbol, Tuple[Type[Strategy], int, dict[str, Any]]] = {
                "SQUID_INK": (SquidInk, 50, {"window_length": 10, "spread": 2}),
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
