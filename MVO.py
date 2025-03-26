import pandas as pd
import numpy as np
import scipy.optimize as sco

class MVOOptimization:

    def __init__(self, rf: float = 0.01, frequency: float = 252, prices: pd.DataFrame = None):

        # 0. Initialize
        self.rf = rf
        self.frequency = frequency
        self.prices = prices

        # 1. Get Prices and Returns
        self.returns = self.prices.pct_change()

        # 2. Calculate Inpus for Optimization
        self.expected_returns = self.returns.mean() * self.frequency
        self.covariance_matrix = self.returns.cov() * self.frequency

        # 3. Optimization settings
        self.number_assets = len(self.prices.columns)
        self.equal_weights = np.ones(self.number_assets) / self.number_assets
        self.bounds = tuple((0, 1) for _ in range(self.number_assets))
        self.cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        pass

    def opt_min_vol(self):

        opt = sco.minimize(
            self.portfolio_volatility,
            self.equal_weights,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.cons
        )

        mv_weights = opt['x']
        mv_sr = -self.neg_sharpe_ratio(mv_weights)
        mv_sd = np.sqrt(np.dot(mv_weights.T, np.dot(self.covariance_matrix, mv_weights)))
        mv_ret = np.sum(self.expected_returns * mv_weights)

        return mv_weights, mv_sr, mv_sd, mv_ret

    def portfolio_volatility(self, weights):
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        return portfolio_std

    def neg_sharpe_ratio(self, weights):
        portfolio_return = np.sum(self.expected_returns * weights)
        portfolio_std = self.portfolio_volatility(weights)
        sharpe_ratio = (portfolio_return - self.rf) / portfolio_std
        return -sharpe_ratio

    def opt_max_sharpe_ratio(self):

        opt = sco.minimize(
            self.neg_sharpe_ratio,
            self.equal_weights,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.cons
        )

        ms_weights = opt['x']
        ms_sr = -self.neg_sharpe_ratio(ms_weights)
        ms_sd = np.sqrt(np.dot(ms_weights.T, np.dot(self.covariance_matrix, ms_weights)))
        ms_ret = np.sum(self.expected_returns * ms_weights)

        return ms_weights, ms_sr, ms_sd, ms_ret

class MVOBacktester:
    def __init__(self,
                 full_prices: pd.DataFrame,
                 rf: float = 0.01,
                 frequency: float = 252,
                 lookback_period: int = 252,
                 rebalancing_period: int = 21,
                 mvo_mode: str = 'max_sharpe'):

        self.full_prices = full_prices
        self.rf = rf
        self.frequency = frequency
        self.lookback_period = lookback_period
        self.rebalancing_period = rebalancing_period
        self.mvo_mode = mvo_mode

    def run_backtest(self) -> tuple[pd.DataFrame, pd.DataFrame]:

        # 0. Initialize
        prices = self.full_prices.copy()
        dates = prices.index

        # 1. Target Definieren (OOS-Returns)
        daily_portfolio_rets = pd.Series(index=prices.index, dtype=float)

        # 2. Loop through all dates
        current_start = 0
        while current_start + self.lookback_period < len(dates):

            # 0. In-Sample-Zeitraum definieren
            in_sample_end = current_start + self.lookback_period
            in_sample_prices = prices.iloc[current_start:in_sample_end]

            # 1. Optimum mit gekürztem Dataframe erhalten
            mvo = MVOOptimization(rf=self.rf,
                                  frequency=self.frequency,
                                  prices=in_sample_prices)

            if self.mvo_mode == 'max_sharpe':
                weights, sr, sd, ret = mvo.opt_max_sharpe_ratio()
            else:
                weights, sr, sd, ret = mvo.opt_min_vol()

            # 2. Out-of-Sample-Zeitraum definieren
            oos_end = in_sample_end + self.rebalancing_period
            # 2.1. Wenn Ende erreicht, dann bis Ende des Dataframes
            if oos_end > len(dates):
                oos_end = len(dates)
            oos_prices = prices.iloc[in_sample_end:oos_end]
            oos_returns = oos_prices.pct_change().dropna()

            print("OOS Period starts at: ", str(oos_prices.head(1).index[0].date()) + " ends at: ", str(oos_prices.tail(1).index[0].date()))


            # 3. Portfolio Performance speichern
            if not oos_returns.empty:
                portfolio_returns = oos_returns.dot(weights)
                daily_portfolio_rets.loc[portfolio_returns.index] = portfolio_returns

            current_start += self.rebalancing_period

        # 3. Drop NaNs
        daily_portfolio_rets.dropna(inplace=True)

        # 4. Create Backtest Dataframe
        bt_tr = pd.DataFrame(daily_portfolio_rets, columns=['BT_RETURN'])
        bt_tr['BT_CUM_RET'] = (1 + bt_tr['BT_RETURN']).cumprod()

        # 5.  Calculate Backtest Performance measures
        avg_return = bt_tr["BT_RETURN"].mean() * self.frequency
        volatility = bt_tr["BT_RETURN"].std() * np.sqrt(self.frequency)
        sharpe_ratio = avg_return / volatility
        roll_max = bt_tr["BT_CUM_RET"].cummax()
        drawdown = bt_tr["BT_CUM_RET"] / roll_max - 1
        max_drawdown = drawdown.min()

        bt_performance = pd.DataFrame({
            "Sharpe Ratio": [sharpe_ratio],
            "Volatility": [volatility],
            "Max Drawdown": [max_drawdown],
            "Average Return": [avg_return]
        }, index=['BT_PERFORMANCE']).T

        return bt_tr, bt_performance


if __name__ == '__main__':

   # 0. Parameter
    rf_ = 0.01
    frequency_ = 252
    lookback_period_ = 252
    rebalancing_period_ = 21
    method_ = 'min_var' # oder 'max_sharpe'

    # 1. Price Data
    prices_ = pd.read_csv(r'G:\TEC101\ALLE\Zink\40_CPF Program\Practice Project\indices_eikon_eod_data.csv', index_col=0,
                         parse_dates=True)
    prices_.index = pd.to_datetime(prices_.index)
    prices_ = prices_.bfill(limit=10).ffill(limit=10)
    prices_ = prices_ / prices_.iloc[0]

    # 2. Optimize over full period
    mvo = MVOOptimization(rf=rf_, frequency=frequency_, prices=prices_)
    mv_weights, mv_sr, mv_sd, mv_ret = mvo.opt_min_vol()
    ms_weights, ms_sr, ms_sd, ms_ret = mvo.opt_max_sharpe_ratio()

    # 3. Backtest initialisieren
    backtester = MVOBacktester(
        full_prices=prices_,
        rf=rf_,
        frequency=frequency_,
        lookback_period=lookback_period_,
        rebalancing_period=rebalancing_period_,
        mvo_mode=method_
    )

    # 4. Run Backtest
    bt_tr, bt_performance = backtester.run_backtest()