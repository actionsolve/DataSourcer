#import datetime
from datetime import datetime, timedelta
import time
import sys
import matplotlib.pyplot as plt
import os
from os.path import isfile

import arrow

import pandas as pd

import requests

from dateutil.parser import parse
from dateutil.tz import gettz
import datetime
from pprint import pprint
import urllib,time,datetime
import sys

# Support libs
sys.path.append('..')
import support.channel_tools as channel_tools
import support.plotting_tools as plot_tools
import support.dataframe_tools as dataframe_tools
import support.time_tools as time_tools

'''
Data Sourcing Libs/Tools to get M1 data from various sources
* Tools to download M1 data + resample (eg M1 -> M5)

Use cases:
1. Download M1 data, for selection of symbols
  Setup symbols  (try with just 2 symbols to test)
  Run:   main() . download_latest_M1_data()

2. Resample downloaded data (eg. 1M -> M5)
  Setup CSV column names and handling
  Run:   main()  . resample_data_file(...
'''

FRAME_INTERVAL        = '1m'

# ------------------------------------------------------
def get_price_data(symbol, start_dt, end_dt, data_interval='1m', output_file_name=None):

    '''
    :param symbol:  eg 'VOD.L' (VOD @LSE, GBP), 'VOD' (VOD @NYSE, USD), 'AAPL', 'RADICO.NS', 'SBIN.NS', 'KPIT.NS'
    :param data_range: eg '1d'
    :param data_interval: can be '1d', '1h', '1m'  1, 15, 60
    :return:

    * The requested range must be within the last 30 days
    * Only 7 days worth of 1m granularity data are allowed to be fetched per request
    '''

    # XXX Check types and ranges
    start_dt = datetime.datetime(start_dt.year, start_dt.month, start_dt.day)
    end_dt = datetime.datetime(end_dt.year, end_dt.month, end_dt.day)

    # https://query1.finance.yahoo.com/v7/finance/chart/RADICO.NS?&interval=1m
    # period1=$START_DATE&period2=$END_DATE
    # "https://query1.finance.yahoo.com/v7/finance/download/$SYMBOL?period1=$START_DATE&period2=$END_DATE&interval=1d&events=history"
    # period1 and period2 are Unix time stamps for your start and end date
    req_text = ('https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d&interval=%s'
                % (symbol, start_dt.timestamp(), end_dt.timestamp(), data_interval))
    print('        [%s]' % req_text)
    res = requests.get(req_text)  # ; print('RES[%s]' % res)
    data = res.json()

    error = data['chart']['error']
    if (str(error) != 'None'):
        print('ERROR[%s]' % error)
        return pd.DataFrame()
    else:
        body = data['chart']['result'][0]
        assert len(body) > 0
        # dt = datetime.datetime
        # dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
        dt = pd.Series(map(lambda x: arrow.get(x).to('GMT').datetime
                           .replace(tzinfo=None), body['timestamp']), name='Datetime')
        df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
        # dg = pd.DataFrame(body['timestamp'])
        #print(df.info())
        df.dropna(inplace=True)  # removing NaN rows
        df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
        df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']      # Renaming columns in pandas
        #print(df.info()); print(df.head(3)); print(df.tail(3))

        # Save
        if(output_file_name != None):
            #df.to_csv(os.path.join(filepath, file_name + '_1D.csv'), float_format='%.6f')
            df.to_csv(output_file_name, float_format='%.6f')

    return df

# ------------------------------------------------------
def download_latest_M1_data(directory_data, symbol_list, column_name_datetime):

    MAX_TOT_DAYS_TO_FETCH = 30
    MAX_DAYS_PER_CALL     = 4

    #latest_end_dt = datetime.datetime(2018, 11, 15, 0, 0)
    latest_end_dt = datetime.datetime.now() #datetime.datetime.date()
    latest_end_dt = latest_end_dt.date()
    limit_start_dt = latest_end_dt - datetime.timedelta(days=MAX_TOT_DAYS_TO_FETCH)
    earliest_actual_start_dt = ""

    #symbol = 'VOD.L'
    #output_file_name = 'Share_%s_%s.csv' % (symbol, interval)
    #data = get_price_data(symbol, start_dt, end_dt, data_interval=interval, output_file_name=output_file_name)

    print('  Fetching %d symbols: %s:  ~%s .. %s,  interval: %s'
          % (len(symbol_list), symbol_list, limit_start_dt, latest_end_dt, FRAME_INTERVAL))
    for symbol in symbol_list:

        # Set timing window limits for next commod
        end_dt = latest_end_dt
        start_dt = end_dt - datetime.timedelta(days=MAX_DAYS_PER_CALL)
        print('\n    Symbol [%s]:  %s .. %s' % (symbol, limit_start_dt, latest_end_dt))

        # Iterate through timing windows
        tot_df = pd.DataFrame()
        while start_dt > limit_start_dt:

            print('      Requesting %s .. %s' % (start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")) )
            #timestamp = "%s_%s" % (start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
            #output_file_name = 'Share_%s_%s_%s.csv' % (symbol, FRAME_INTERVAL, timestamp)
            #print('        -> File [%s]' % output_file_name)
            #earliest_actual_start_dt = start_dt

            tmp_df = get_price_data(symbol, start_dt, end_dt, data_interval=FRAME_INTERVAL)
            #                        , output_file_name=output_file_name)
            print('        Fetched %d rows, %d cols (%s)'
                  % (len(tmp_df), len(tmp_df.columns), tmp_df.columns.values))
            if(len(tmp_df)==0):
                break

            # Append to df
            tot_df = pd.concat([tmp_df, tot_df], axis=0)

            # Increment timing window
            end_dt = end_dt - datetime.timedelta(days=MAX_DAYS_PER_CALL)
            start_dt = end_dt - datetime.timedelta(days=MAX_DAYS_PER_CALL)


        # Save concatenated CSV
        print('\n      Consolidated to %d rows, %d cols (%s)'
                % (len(tot_df), len(tot_df.columns), tot_df.columns.values))
        #timestamp = "%s_%s" % (earliest_actual_start_dt.strftime("%Y%m%d"), latest_end_dt.strftime("%Y%m%d"))
        #output_file_name = 'Share_%s_%s_Tot_%s.csv' % (symbol, FRAME_INTERVAL, timestamp)
        #output_file_name = 'Share_%s_%s_Tot_%s.csv' % (symbol, FRAME_INTERVAL, timestamp)
        #print('        Writing to: [%s]' % output_file_name)
        #tot_df.to_csv(output_file_name, float_format='%.6f')

        # If file exists, concat (preference to latest data), else write new
        #output_file_name = 'Share_%s_%s.csv' % (symbol, FRAME_INTERVAL)
        output_file_name = os.path.join(directory_data, 'Share_%s_%s.csv' % (symbol, FRAME_INTERVAL))
        if(isfile(output_file_name)):

            #print('        Writing to: [%s]' % ('Recn_' +output_file_name))         # XXX FIX
            #tot_df.to_csv('Recn_'+output_file_name, float_format='%.6f')            # XXX REMOVE XXXXXXXXXXXXXX
            print('        Reading previous: [%s]' % output_file_name)
            df_prev = pd.read_csv(output_file_name)

            # Ensure datetime index
            df_prev[column_name_datetime] = pd.to_datetime(df_prev[column_name_datetime])
            df_prev = df_prev.set_index(column_name_datetime);      # ; print(df_prev.info())

            tot_df = tot_df.combine_first(df_prev)

        print('        Writing to: [%s]' % output_file_name)
        tot_df.to_csv(output_file_name, float_format='%.6f')


    '''
    Checks
      Load in in XLS
      Insert plots
        'datetime':  expect steps for days and weekends
        'CLOSE':  expect continuous brownian
    '''


# ---------------------------------------------------------------------
def resample_m1_csv(input_file_name, frames_per_min      #
        , column_name_datetime                      # Usually 'Date', 'Datetime' or similar
        , resample_column_handling_dict
        , saved_column_order
        , output_file_name
        , drop_na=True                              # default to NOT creating artificial frames
        , verbose=False):
    '''
    Change timeframe,
        minutes_per_frame=1440 -> 1 frame for every day

    Example 'resample_column_handling_dict'
            resample_column_handling_dict = {
                'Open'  : 'first',
                'High'  : 'max',
                'Low'   : 'min',
                'Close' : 'last'  # ,'Volume':'sum'
                }
    XXX Todo
      Sanity checks
      Check datetime_column_name and other  columns present
      Check M1 data
      Fix column order of output df, based on keys from resample_column_handling_dict
        df = df[['O','H','L','C']]
    '''

    if(verbose):
        print('      Resampling(%s) @ %d frames orig frames -> each new frame, keys: %s'
                % (input_file_name, frames_per_min, list(resample_column_handling_dict.keys())))

    # Load file
    df_orig = pd.read_csv(input_file_name)
    #date_parser = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d-%H-%M-%S')
    #date_parser = lambda s: datetime.datetime.strptime(s, date_format)
    #df = pd.read_csv(file_name, nrows=(num_frames+1), date_parser=date_parser)
    #print(df.info())

    if (verbose):
        print('        Read in: %d frames -> %d columns: %s'
              % (len(df_orig), len(df_orig.columns), df_orig.columns.values ))

    # Check datetime index
    df_orig[column_name_datetime] = pd.to_datetime(df_orig[column_name_datetime])
    df_orig = df_orig.set_index(column_name_datetime)
    #print(df.info())                                                # XXX

    # Resample
    num_frames_start = len(df_orig)
    resample_str = str(frames_per_min) + 'T'
    df_resampled = df_orig.resample(resample_str, closed='left', label='right').apply(resample_column_handling_dict)
    if (verbose): print('        Resampled: %d -> %d frames' % (num_frames_start, len(df_resampled)))

    # Drop rows containing NaN
    if(drop_na):
        df_resampled.dropna(inplace=True)
        if(verbose): print('        Dropped artificial frames -> %d frames' % (len(df_resampled)))

    # Write
    if (verbose):
        print('        Saving to:  [%s]    using cols: %s' % (output_file_name, str(saved_column_order)))
    df_resampled = df_resampled[list(saved_column_order)]
    df_resampled.to_csv(output_file_name, float_format='%.6f')

    # Return df
    return df_resampled


# ------------------------------------------------------
def resample_data_file(directory_data, symbol, frame_size_in_minutes
        , resample_column_handling_dict
        , saved_column_order, column_name_datetime
        , show_plots=False):
    '''
    Share_AV.L_1m.csv
    Share_BP.L_1m.csv
    Share_GLEN.L_1m.csv
    Share_LLOY.L_1m.csv
    Share_MRW.L_1m.csv
    Share_NG.L_1m.csv
    Share_VOD.L_1m.csv
    '''
    #input_file_name  = 'Share_VOD.L_1m.csv'
    input_file_name  = os.path.join(directory_data, 'Share_%s_%s.csv' % (symbol, FRAME_INTERVAL))
    #output_file_name = 'Share_%s_%dm.csv' % (symbol, frame_size_in_minutes)
    output_file_name = os.path.join(directory_data, 'Share_%s_%dm.csv' % (symbol, frame_size_in_minutes))

    #resample_column_handling_dict = {
    #    'OPEN': 'first',
    #    'HIGH': 'max',
    #    'LOW': 'min',
    #    'CLOSE': 'last'  # ,'Volume':'sum'
    #}
    #datetime_column_name = 'Datetime'

    # Resample
    df3 = resample_m1_csv(input_file_name, frame_size_in_minutes
            , column_name_datetime, resample_column_handling_dict, saved_column_order
            , output_file_name, drop_na=True, verbose=True)
    #print(df3.info()) ; print(df3.head(2))


    if(show_plots):
        # Load original
        df1 = pd.read_csv(input_file_name)
        df1['Date'] = pd.to_datetime(df1[column_name_datetime]) ; df1 = df1.set_index(column_name_datetime)
        df1 = df1.drop(['VOLUME'], axis=1)  ; df1 = df1.drop(['Date'], axis=1)

        # Plot comparisons
        axes = df1.plot(subplots=True, grid=True)  # array ax for each plot
        axes[0].set_title(input_file_name)
        [ax.grid(True, which='minor') for ax in axes]
        plt.gcf().set_size_inches(6, 10) ; plt.tight_layout() #; plt.grid()

        #  Load resampled
        df2 =  pd.read_csv(output_file_name)
        df2['Date'] = pd.to_datetime(df2[column_name_datetime]) ; df2 = df2.set_index(column_name_datetime)
        #df2 = df2.drop(['VOLUME'], axis=1)  ;
        df2 = df2.drop(['Date'], axis=1)

        # Plot comparisons
        axes = df2.plot(subplots=True, grid=True)  # array ax for each plot
        axes[0].set_title(output_file_name)
        [ax.grid(True, which='minor') for ax in axes]
        plt.gcf().set_size_inches(6, 10) ; plt.tight_layout() #; plt.grid()

        # Keep visible
        plt.show(block=True)



# -------------------------------------------------------------------
def resample_M1_data(directory_data, symbol_list, column_name_datetime, frame_sizes_in_minutes):

    # Specify CSV columns and resampling
    resample_column_handling_dict = {
        'OPEN'    : 'first',
        'HIGH'    : 'max',
        'LOW'     : 'min',
        'CLOSE'   : 'last',
        'VOLUME'  : 'sum'
    }                                                         # N.B. Does NOT preserve key order
    saved_column_order = ('OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME')
    # print(resample_column_handling_dict.keys())  ; print(saved_column_order)
    #resampling_frames_per_min_vals = (5, 10, 20, 60)

    # Resample each symbol, for each frames_per_min
    for symbol in symbol_list:

        for frame_size in frame_sizes_in_minutes:

            resample_data_file(directory_data, symbol, frame_size, resample_column_handling_dict
                    , saved_column_order, column_name_datetime
                    , show_plots=False)
                    #, show_plots=(frames_per_min==60))


def _get_symbols():

    # Vod -> VOD:LN -> VOD.L,
    # FTSE100 -> UKX:IND , UKX.L , ^FTSE.L ->

    # https://uk.finance.yahoo.com/world-indices
    # FTSE 100: '^FTSE'
    # FTSE 250: ^FTMC
    # S&P 500:  ^GSPC

    #  2018-11-23  Symbols no longer available: https://en.wikipedia.org/wiki/FTSE_100_Index
    # symbol_list.extend(('BSB', 'AAA'))

    symbol_list = []
    debug_only = False

    if debug_only:
        # Currencies
        symbol_list.extend(('GLEN.L', 'MRW.L'))   # 2018 active

        # TEST ONLY
        #symbol_list.extend(['AAL.L'])     # test only

    else:
        # Currencies
        symbol_list.extend(('GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCHF=X', 'USDCAD=X'))

        # Indices
        symbol_list.extend(('^FTSE', '^FTMC', '^N100', '^RUT', '^GSPC', '^FCHI', '^N225', '^AORD', '^JN0U.JO'))

        # Shares
        symbol_list.extend(('GLEN.L', 'MRW.L'))   # 2018 active
        symbol_list.extend(('NG.L', 'BP.L', 'LLOY.L', 'AV.L', 'VOD.L', 'TSCO.L')) # 2018 active
        symbol_list.extend(('BLND.L', 'DLG.L', 'GVC.L', 'RMG.L', 'SGE.L', 'WPP.L'))  # 2018 risers
        symbol_list.extend(('AAL.L', 'EVR.L', 'FRES.L', 'RDSB.L', 'RIO.L', 'WG.L'))  # 2018 fallers




    return symbol_list


# -------------------------------------------------------------------
def main():

    column_name_datetime = 'Datetime'
    directory_data       = 'Data'

    # Get symbols
    symbol_list = _get_symbols()
    for a in symbol_list: print('Fetching: %s' % a)
    assert len(set(symbol_list)) == len(symbol_list)    # No duplicates


    # Download latest M1 data, and consolidates with any existing
    if(False):
        download_latest_M1_data(directory_data, symbol_list, column_name_datetime)


    # Resample M1 data
    if(True):
        #frame_sizes_in_minutes = [5]
        frame_sizes_in_minutes = (5, 10, 20, 60)
        resample_M1_data(directory_data, symbol_list, column_name_datetime, frame_sizes_in_minutes)



# -------------------------------------------------------------------
if __name__ == "__main__":

    print("\n%s Started" % time_tools.get_timestamp())
    start_time = time.time()  # Start the clock

    pd.set_option('display.expand_frame_repr', False)  # Ensure columns do NOT wrap on page

    main()

    # unittests

    print('%s Finished    (Elapsed: %.3f)'
          % (time_tools.get_timestamp(), (time.time() - start_time)))
    exit(0)

