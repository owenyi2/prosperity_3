import json
from typing import Any, Dict, Tuple, Type, Literal, Optional

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np
from statistics import NormalDist

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
    
    best_sell_pr = list(filter(lambda x: x > fair_value, osell.keys()))[0]
    best_buy_pr = list(filter(lambda x: x < fair_value, obuy.keys()))[0]

    undercut_buy = best_buy_pr + 1
    undercut_sell = best_sell_pr - 1
    
    cpos = position
    for ask, vol in osell.items(): # vol < 0
        if ((ask < fair_value) or ((position < 0) and (ask == fair_value))):
            if cpos < position_limit:
                order_for = min(-vol, position_limit - cpos)
                cpos += order_for
                strat.buy(ask, order_for)

    if cpos < position_limit:
        num = position_limit - cpos
        our_bid = min(undercut_buy, fair_value - spread)
        if position == -position_limit:
            our_bid = fair_value - spread // 2 
            # when we hit a position_limit, we want to be more aggressive with our bids/asks in order to liquidate position faster
        strat.buy(our_bid, num) 
        cpos += num

    cpos = position
    for bid, vol in obuy.items(): # vol > 0
        if ((bid > fair_value) or ((position > 0) and (bid == fair_value))):
            if cpos > -position_limit:
                order_for = min(vol, position_limit - cpos)
                cpos -= order_for
                strat.sell(bid, order_for)

    if cpos > -position_limit:
        num = position_limit - (-cpos)
        our_ask = max(undercut_sell, fair_value + spread)
        if position == position_limit:
            our_ask = fair_value + spread // 2
        strat.sell(our_ask, num)
        cpos -= num
     
class ForecastStrategy(Strategy):
    def get_midprice(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2
    
    def reach_position(self, position: int, desired_position: int, order_depth: OrderDepth, price: int) -> None:
        change = abs(desired_position - position)

        if desired_position > position:
            self.buy(price, change)

        if desired_position < position:
            self.sell(price, change)

def bisection(f, tol):
    a = 0.00001
    b = 0.02

    if f(a) * f(b) >= 0:
        return 0

    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        f_mid = f(midpoint)

        if abs(f_mid) < tol:  # Found exact root
            return midpoint
        elif f(a) * f_mid < 0:  # Root is in the left half
            b = midpoint
        else:  # Root is in the right half
            a = midpoint

    return (a + b) / 2  # Return the midpoint as the root approximation

def trader_BS_CALL(S, K, T, sigma):
    N = NormalDist().cdf
    if sigma == 0:
        sigma = 1e-12
    d1 = (np.log(S/K) + (sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * N(d1) - K * N(d2)

def trader_implied_vol(opt_value, S, K, T):
    tol = 10**-8
    return bisection(lambda sigma: trader_BS_CALL(S, K, T, sigma) - opt_value, tol)

def trader_delta(S, K, sigma, T):
    N = NormalDist().cdf
    if sigma == 0:
        sigma = 1e-12
    return N((np.log(S/K) + (sigma**2/2)*T) / (sigma*np.sqrt(T)))

class OptionChain():
    def __init__(self, day: int,
                 spread_centre: list[float],
                 spread_dispersion: list[float]
                 ):
        self.strikes = [9500, 9750, 10000, 10250, 10500]
        self.delta: List[Optional[float]] = [None] * len(self.strikes)
        self.day = day
        self.midprice: list[Optional[float]] = [None] * (len(self.strikes) + 1)
        self.spread_centre: list[float] = spread_centre
        self.spread_dispersion: list[float] = spread_dispersion
    
    def compute_midprices(self, state: TradingState, previous_midprice: list[float]) -> None:
        self.midprice = previous_midprice 
        for i in range(len(self.strikes) + 1):
            if i == len(self.strikes):
                symbol = "VOLCANIC_ROCK"
            else:
                symbol = f"VOLCANIC_ROCK_VOUCHER_{self.strikes[i]}"

            order_depth = state.order_depths[symbol]
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depth.sell_orders.items())

            if len(buy_orders) != 0 and len(sell_orders) != 0:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                self.midprice[i] = (popular_buy_price + popular_sell_price) / 2
            else:
                pass # keep previous_midprice

                # if len(buy_orders) != 0: # missing current sell_order
                #     popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                #     self.midprice[symbol] = (self.midprice[symbol] + popular_buy_price)/2

                # if len(sell_orders) != 0: # missing current bid_order
                #     popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                #     self.midprice[symbol] = (self.midprice[symbol] + popular_sell_price)/2
             
    def compute_deltas(self, state: TradingState) -> None:
        for i in range(len(self.strikes)):
            K = self.strikes[i]
            opt_value = self.midprice[i]
            S = self.midprice[-1]
            T = 7 - (self.day + state.timestamp  / 1_000_000)
            
            if opt_value is None or S is None:
                self.delta[i] = None
                continue

            sigma = trader_implied_vol(opt_value, S, K, T)
            self.delta[i] = trader_delta(S, K, sigma, T)
     
class VolcanicRock(ForecastStrategy):
    def __init__(self, symbol: str, position_limit: int, 
                 option_chain: OptionChain,
                 ):
        super().__init__(symbol, position_limit)
        self.option_chain = option_chain
        self.previous_midprices = [None] * 6

    def load_state(self, state: list[Optional[float]]) -> None:
        self.previous_midprices = state

    def save_state(self) -> list[Optional[float]]:
        return self.previous_midprices
     
    def act(self, state: TradingState) -> None:
        self.option_chain.compute_midprices(state, self.previous_midprices) 
        self.option_chain.compute_deltas(state)

        position: int = state.position.get(self.symbol, 0)
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        # # for now we delta-hedge a single option strike
        # i = 0
        # if (self.option_chain.midprice[i] is None or 
        #     self.option_chain.midprice[-1] is None or 
        #     self.option_chain.delta[i] is None
        #     ):
        #     return  
        #     
        # spread = self.option_chain.midprice[-1] - self.option_chain.delta[i] * self.option_chain.midprice[i]
        # score = np.clip((spread - self.option_chain.spread_centre[i]) / self.option_chain.spread_dispersion[i], -1, 1)

        # desired_position = int(round(-score * 200))
        # price = int(round(self.option_chain.midprice[-1]))
    
        # # print(f"delta: {self.option_chain.delta[i]}")
        # # print(f"spread: {spread}")
        # # print(f"score: {score}")
        # # print(f"desired_position: {desired_position}")
        # # input()
        # self.reach_position(position, desired_position, order_depth, price)

class VolcanicRockVoucher(ForecastStrategy):
    def __init__(self, symbol: str, position_limit: int, 
                 option_chain: OptionChain,
                 i: int
                 ):
        super().__init__(symbol, position_limit)
        self.option_chain = option_chain
        self.i = i

    def act(self, state: TradingState) -> None:
        position: int = state.position.get(self.symbol, 0)
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        # for now we delta-hedge a single option strike
        i = 0
        if (self.option_chain.midprice[i] is None or 
            self.option_chain.midprice[-1] is None or 
            self.option_chain.delta[i] is None
            ):
            return  
            
        spread = self.option_chain.midprice[-1] - self.option_chain.delta[i] * self.option_chain.midprice[i]
        score = np.clip((spread - self.option_chain.spread_centre[i]), -self.option_chain.spread_dispersion[i], self.option_chain.spread_dispersion[i]) / self.option_chain.spread_dispersion[i]
        # TODO: we are calculating the score incorectly, we are supposed to clip before diviiding
        # print(spread)
        # print(spread - self.option_chain.spread_centre[i])
        # print(score)
        # input()

        desired_position = int(round(score * self.option_chain.delta[i] * 200))
        
        price = int(round(self.option_chain.midprice[-1]))
    
        self.reach_position(position, desired_position, order_depth, price)


class Trader:
    def __init__(self) -> None:
        option_chain = OptionChain(day = 0, 
                                   spread_centre = [9503, 9751], 
                                   spread_dispersion = [2, 2])

        init_dict: dict[Symbol, Tuple[Type[Strategy], int, dict[str, Any]]] = {
                "VOLCANIC_ROCK": (VolcanicRock, 400, {
                    "option_chain": option_chain,
                    }), # VOLCANIC ROCK MUST OCCUR IN THIS DICT BEFORE ANY OTHER VOLCANIC ROCK VOUCHER BECUASE IT IS REPONSIBLE FOR PRECOMPUTING VALUES SUCH AS OPTION DELTA ik its kinda fragile
                "VOLCANIC_ROCK_VOUCHER_9500": (VolcanicRockVoucher, 200, {
                    "option_chain": option_chain,
                    "i": 0
                    }), # sigh we need to fix this, the position is hard stuck on 100%
                # "VOLCANIC_ROCK_VOUCHER_9750": (VolcanicRockVoucher, 200, {
                #     "option_chain": option_chain,
                #     "i": 1
                #     }),
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
