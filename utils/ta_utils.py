import numpy as np
import pandas as pd
from IPython.core.display import display, HTML

import yfinance as yf
from ta import *

from utils.ta_utils import *

from bokeh.plotting import figure, show
from bokeh.layouts import row
from bokeh.io import output_notebook

def get_dataset_for_stock(stock_name):
    """ Get Pandas DataFrame for particular stock from
        Yahoo Finance API.
        
    Parameters
    ----------
    stock_name : string
        Yahoo Finance stock symbol (example: NVDA).
        
    Returns
    -------
    df : Pandas DataFrame instance.
        Data with day index, prices and volume.
    
    """
    df = yf.Ticker(stock_name).history(period="max")
    
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]
    
    return df


def show_stock_analysis(df):
    """ Generate stock summary standard description.
    
    Parameters
    ----------
    df : Pandas DataFrame instance.
        Input data from stock.
    
    """
    
    def get_hist_from_price_diff(price_series, days_window):
        future_price = price_series.shift(-days_window)
        diff = future_price - price_series

        hist_values, edges = np.histogram(diff.fillna(0).values, density=True, bins=50)
        
        p33, p66 = np.quantile(diff.fillna(0).values, [0.33, 0.66])
        
        mean, std = np.mean(diff.fillna(0).values), np.std(diff.fillna(0).values)

        # \n P33={} | P66={} | Mean={} | Std={}
        p = figure(title='Price diffs histogram ({}d window).'.format(
                        days_window,
                        #round(p33, 3),
                        #round(p66, 3),
                        #round(mean, 3),
                        #round(std, 3),
                   background_fill_color="#fafafa"),
                   plot_width=350, 
                   plot_height=350)
        
        p.quad(top=hist_values, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white", alpha=0.5)

        return p
    
    def get_candles_plot(df):
        from math import pi
        
        inc = df.close > df.open
        dec = df.open > df.close
        w = 12*60*60*1000 # half day in ms

        p = figure(x_axis_type="datetime", plot_width=900, title = "")
        p.xaxis.major_label_orientation = pi/4
        p.grid.grid_line_alpha=0.3

        p.segment(df.index, df.high, df.index, df.low, color="black")
        p.vbar(df.index[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
        p.vbar(df.index[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")
        
        return p
    
    display(HTML("<h2>Resumen de los datos</h2>"))
    display(df.describe())
    
    display(HTML("<h2>Rango de fechas</h2>"))
    display(df.index.min(), df.index.max())
    
    display(HTML("<h2>Gráfico de velas</h2>"))
    
    show(get_candles_plot(df))
    
    display(HTML("<h2>Distribución de diferenciales de precio para distintas ventanas de tiempo</h2>"))
    
    plots_row = []

    for day_window in [5, 10, 30]:
        p = get_hist_from_price_diff(df.close, day_window)
        plots_row.append(p)
        
    show(row(plots_row))
    

def add_technical_indicators(df, indicators=[]):
    """ Add technical indicators for particular dataset.
    
    Parameters
    ----------
    df : Pandas DataFrame instance.
        Input data from stock.
    
    indicators : list of string elements.
        Technical indicators names to include. Default include all.
    
    Returns
    -------
    df : Pandas DataFrame instance.
        Original dataset with added indicators.
        
    """
    
    include_all_indicators = True
    
    if indicators != []:
        include_all_indicators = False 
    
    if ("MACD" in indicators) or include_all_indicators:
        df["ti_macd"] = trend.macd(close=df.close, 
                                   n_fast=12,
                                   n_slow=26, 
                                   fillna=True)
        df["ti_macd_diff"] = df.close - df.ti_macd
        df["ti_macd_dm1"] = df.ti_macd.shift(1).fillna(0)
        df["ti_macd_dm2"] = df.ti_macd.shift(2).fillna(0)
        
    if ("RSI" in indicators) or include_all_indicators:
        df["ti_rsi"] = momentum.rsi(close=df.close, 
                                    n=14, 
                                    fillna=True)
        df["ti_rsi_overbought"] = df.ti_rsi > 70
        df["ti_rsi_oversold"] = df.ti_rsi < 30
        
    if ("CMF" in indicators) or include_all_indicators:
        df["ti_cmf"] = volume.chaikin_money_flow(high=df.high, 
                                                 low=df.low, 
                                                 close=df.close, 
                                                 volume=df.volume, 
                                                 n=20, 
                                                 fillna=True)
        df["ti_cmf_dm1"] = df.ti_cmf.shift(1).fillna(0)
        df["ti_cmf_dm2"] = df.ti_cmf.shift(2).fillna(0)
        
    if ("VPI" in indicators) or include_all_indicators:
        df["ti_vpi"] = volume.volume_price_trend(close=df.close, 
                                                 volume=df.volume)
        
    if ("BB" in indicators) or include_all_indicators:
        df["ti_bb"] = volatility.bollinger_hband_indicator(close=df.close, 
                                                           n=20, 
                                                           ndev=2, 
                                                           fillna=True)
    
    if ("ATR" in indicators) or include_all_indicators:
        df["ti_atr"] = volatility.average_true_range(high=df.high, 
                                                     low=df.low, 
                                                     close=df.close, 
                                                     n=14, 
                                                     fillna=True)
        
    if ("IC" in indicators) or include_all_indicators:
        df["ti_ic"] = trend.ichimoku_a(high=df.high, 
                                       low=df.low, 
                                       n1=9, 
                                       n2=26, 
                                       visual=False, 
                                       fillna=True)
        
    if ("MFI" in indicators) or include_all_indicators:
        df["ti_mfi"] = momentum.money_flow_index(high=df.high, 
                                                 low=df.low, 
                                                 close=df.close, 
                                                 volume=df.volume, 
                                                 n=14, 
                                                 fillna=True)
        
    if ("CR" in indicators) or include_all_indicators:
        df["ti_cr"] = others.daily_log_return(close=df.close,
                                              fillna=True)
    
    return df


def plot_features_relevance(features, relevances):
    """ Plot feature relevance in bar chart.
    
    Parameters
    ----------
    features : List of strings.
        Features names.
    
    relevances : List of floats.
        Relevance for each feature.
    """
    
    assert len(features) == len(relevances)
    
    p = figure(x_range=features, plot_width=900, plot_height=250)
    p.vbar(x=features, top=relevances, width=0.9)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = "vertical"

    show(p)
    

def plot_time_windows_pnls(pnls, time_windows):
    """ Plot time windows PNLs.
    
    Parameters
    ----------
    pnls : List of float.
        PNL for each time_window.
    
    time_windows : List of int.
        Represent each window tested.
    """

    p = figure(x_range=[str(t) for t in time_windows], plot_width=600, plot_height=250)
    p.vbar(x=[str(t) for t in time_windows], top=pnls, width=0.9)

    show(p)


def show_strategy_actions(df):
    from math import pi

    p = figure(plot_width=900, plot_height=300, x_axis_type="datetime")

    # add a line renderer
    p.line(df.index, df.close, line_width=2)

    df_buy = df[df.order == 'buy']
    p.triangle(df_buy.index, df_buy.close, size=10, color="green", alpha=0.5)

    df_sell = df[df.order == 'sell']
    p.triangle(df_sell.index, df_sell.close, size=10, color="red", angle=pi, alpha=0.5)

    show(p)