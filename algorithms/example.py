import json
from typing import Any, Dict

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import pprint
from collections import OrderedDict

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


class Trader:
    position_limits = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50
            }

    def market_take(self, symbol: Symbol, fair_price: int, order_depth: OrderDepth, position: int):
        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        
        # pprint.pp(osell) 
        # pprint.pp(obuy) 

        position_limit = self.position_limits[symbol]
        orders = []
        

        cpos = position
        for ask, vol in osell.items():
            if ((ask < fair_price) or ((position < 0) and (ask == fair_price))) and cpos < position_limit:
                order_for = min(-vol, position_limit - position)
                cpos += order_for
                orders.append(Order(symbol, ask, order_for))
        
        cpos = position
        for bid, vol in obuy.items():
            if ((bid > fair_price) or ((position > 0) and (bid == fair_price))) and cpos > -position_limit:
                order_for = max(-vol, -position_limit - cpos)
                cpos += order_for
                orders.append(Order(symbol, bid, order_for))
        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        trader_data = ""
        
        symbol = "RAINFOREST_RESIN"
        result[symbol] = self.market_take(
                symbol, 
                10000, 
                state.order_depths[symbol], 
                state.position.get(symbol, 0)
            )

        """
        INFO: 
            Ignore conversions for now. This mechanic is used later

            trader_data: str is your only way to store state between ticks

            result: dict[Symbol, list[Order]] are the orders to submit
        """
        
        # pprint.pp(json.loads(
        #     state.toJSON()
        #     ))
        # input()

             

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
