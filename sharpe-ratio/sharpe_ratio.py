import pandas as pd
import numpy as np
import time

def read_csvs_batters():
    paths = [
            '../projections/hitters/atc.csv',
            '../projections/hitters/bat.csv',
            '../projections/hitters/bat_x.csv',
            '../projections/hitters/depth_charts.csv',
            '../projections/hitters/oopsy.csv',
            '../projections/hitters/steamer.csv',
            '../projections/hitters/zips.csv',
            '../projections/hitters/zips_dc.csv'
            ]

    dfs = pd.concat([pd.read_csv(path, delimiter = '\t') for path in paths], ignore_index = True)

    dfs_sharpe = dfs[["Name", "HR", "R", "RBI", "SB", "OBP"]].groupby("Name").agg(['mean', 'std']).reset_index()
    adps = np.array([np.mean(dfs[dfs['Name'] == name]['ADP'].values) for name in dfs_sharpe['Name']])
    dfs_sharpe_roster = dfs_sharpe[adps != 999]
    dfs_sharpe_replacement = dfs_sharpe[adps == 999]

    replacement_level_player_hr = np.percentile(dfs_sharpe_replacement['HR']['mean'], 95)
    replacement_level_player_r = np.percentile(dfs_sharpe_replacement['R']['mean'], 95)
    replacement_level_player_rbi = np.percentile(dfs_sharpe_replacement['RBI']['mean'], 95)
    replacement_level_player_sb = np.percentile(dfs_sharpe_replacement['SB']['mean'], 95)
    replacement_level_player_obp = np.percentile(dfs_sharpe_replacement['OBP']['mean'], 95)

    replacement_level_player = {
                                   "HR": replacement_level_player_hr,
                                   "R": replacement_level_player_r,
                                   "RBI": replacement_level_player_rbi,
                                   "SB": replacement_level_player_sb,
                                   "OBP": replacement_level_player_obp
                               }
    
    for category in ['HR', 'R', 'RBI', 'SB', 'OBP']:
        dfs_sharpe[f'{category}_sharpe'] = (dfs_sharpe[category]['mean'] - replacement_level_player[category])/(dfs_sharpe[category]['std'])
        dfs_sharpe[f'{category}_sharpe'].fillna(dfs_sharpe[category]['mean'])
        dfs_sharpe[f'{category}_sharpe'] = dfs_sharpe[f'{category}_sharpe'].mask(np.isinf(dfs_sharpe[f'{category}_sharpe']), dfs_sharpe[category]['mean'])

    dfs_sharpe['Total_sharpe'] = dfs_sharpe['HR_sharpe'] + dfs_sharpe['R_sharpe'] + dfs_sharpe['RBI_sharpe'] + dfs_sharpe['SB_sharpe'] + dfs_sharpe['OBP_sharpe']

    pd.set_option('display.max_rows', None)

    print('Top Players By Risk Adjusted Home Runs')
    print(dfs_sharpe.sort_values(by = 'HR_sharpe', ascending = False).head(10))
    
    print('Top Players By Risk Adjusted Runs')
    print(dfs_sharpe.sort_values(by = 'R_sharpe', ascending = False).head(10))
    
    print('Top Players By Risk Adjusted RBIs')
    print(dfs_sharpe.sort_values(by = 'RBI_sharpe', ascending = False).head(10))
    
    print('Top Players By Risk Adjusted Stolen Bases')
    print(dfs_sharpe.sort_values(by = 'SB_sharpe', ascending = False).head(10))
    
    print('Top Players By Risk Adjusted OBP')
    print(dfs_sharpe.sort_values(by = 'OBP_sharpe', ascending = False).head(10))

    print('Top Risk Adjusted Players')
    print(dfs_sharpe.sort_values(by = 'Total_sharpe', ascending = False).head(240))

    #print(dfs_sharpe[dfs_sharpe['Name'] == 'Aaron Judge'])

    return dfs_sharpe

def read_csvs_pitchers(starters = True):
    paths = [
            '../projections/pitchers/atc.csv',
            '../projections/pitchers/bat.csv',
            '../projections/pitchers/bat_x.csv',
            '../projections/pitchers/depth_charts.csv',
            '../projections/pitchers/oopsy.csv',
            '../projections/pitchers/steamer.csv',
            '../projections/pitchers/zips.csv',
            '../projections/pitchers/zips_dc.csv'
            ]

    if starters:
        dfs = pd.concat([pd.read_csv(path, delimiter = '\t') for path in paths], ignore_index = True)
        dfs = dfs[dfs['GS'] > 5]

        dfs_sharpe = dfs[["Name", "W", "SO", "WHIP", "ERA"]].groupby("Name").agg(['mean', 'std']).reset_index()
        adps = np.array([np.mean(dfs[dfs['Name'] == name]['ADP'].values) for name in dfs_sharpe['Name']])
        dfs_sharpe_roster = dfs_sharpe[adps != 999]
        dfs_sharpe_replacement = dfs_sharpe[adps == 999]

        replacement_level_player_w = np.percentile(dfs_sharpe_replacement["W"]['mean'], 95)
        replacement_level_player_k = np.percentile(dfs_sharpe_replacement["SO"]['mean'], 95)
        replacement_level_player_whip = np.percentile(dfs_sharpe_replacement["WHIP"]['mean'], 5)
        replacement_level_player_era = np.percentile(dfs_sharpe_replacement["ERA"]['mean'], 5)

        replacement_player = {
                                 "W": replacement_level_player_w,
                                 "SO": replacement_level_player_k,
                                 "WHIP": replacement_level_player_whip,
                                 "ERA": replacement_level_player_era
                             }
        
        for category in ['W', 'SO', 'WHIP', 'ERA']:
            dfs_sharpe[f'{category}_sharpe'] = (dfs_sharpe[category]['mean'] - replacement_player[category])/(dfs_sharpe[category]['std'])
            dfs_sharpe[f'{category}_sharpe'].fillna(dfs_sharpe[category]['mean'])
            dfs_sharpe[f'{category}_sharpe'] = dfs_sharpe[f'{category}_sharpe'].mask(np.isinf(dfs_sharpe[f'{category}_sharpe']), dfs_sharpe[category]['mean'])


        dfs_sharpe['Total_sharpe'] = dfs_sharpe['W_sharpe'] + dfs_sharpe['SO_sharpe'] - dfs_sharpe['WHIP_sharpe'] - dfs_sharpe['ERA_sharpe']

        pd.set_option('display.max_rows', None)

        print('Top Players By Risk Adjusted Wins')
        print(dfs_sharpe.sort_values(by = 'W_sharpe', ascending = False).head(10))
        
        print('Top Players By Risk Adjusted Strikeouts')
        print(dfs_sharpe.sort_values(by = 'SO_sharpe', ascending = False).head(10))
        
        print('Top Players By Risk Adjusted WHIP')
        print(dfs_sharpe.sort_values(by = 'WHIP_sharpe', ascending = True).head(10))
        
        print('Top Players By Risk Adjusted ERA')
        print(dfs_sharpe.sort_values(by = 'ERA_sharpe', ascending = True).head(10))

        print('Top Risk Adjusted Players')
        print(dfs_sharpe.sort_values(by = 'Total_sharpe', ascending = False).head(240))

        return dfs_sharpe
    else:
        dfs = pd.concat([pd.read_csv(path, delimiter = '\t') for path in paths], ignore_index = True)
        dfs = dfs[dfs['G'] - dfs['GS'] > 5]
        dfs["SVHLD"] = dfs['SV'] + dfs['HLD']

        dfs_sharpe = dfs[["Name", "W", "SO", "SVHLD", "WHIP", "ERA"]].groupby("Name").agg(['mean', 'std']).reset_index()
        adps = np.array([np.mean(dfs[dfs['Name'] == name]['ADP'].values) for name in dfs_sharpe['Name']])
        dfs_sharpe_roster = dfs_sharpe[adps != 999]
        dfs_sharpe_replacement = dfs_sharpe[adps == 999]

        replacement_level_player_w = np.percentile(dfs_sharpe_replacement["W"]['mean'], 95)
        replacement_level_player_sv = np.percentile(dfs_sharpe_replacement["SVHLD"]['mean'], 95)
        replacement_level_player_k = np.percentile(dfs_sharpe_replacement["SO"]['mean'], 95)
        replacement_level_player_whip = np.percentile(dfs_sharpe_replacement["WHIP"]['mean'], 5)
        replacement_level_player_era = np.percentile(dfs_sharpe_replacement["ERA"]['mean'], 5)
        
        replacement_player = {
                                 "W": replacement_level_player_w,
                                 "SO": replacement_level_player_k,
                                 "WHIP": replacement_level_player_whip,
                                 "ERA": replacement_level_player_era,
                                 "SVHLD": replacement_level_player_sv
                             }
        
        for category in ['W', 'SO', 'WHIP', 'ERA', 'SVHLD']:
            dfs_sharpe[f'{category}_sharpe'] = (dfs_sharpe[category]['mean'] - replacement_player[category])/(dfs_sharpe[category]['std'])
            dfs_sharpe[f'{category}_sharpe'].fillna(dfs_sharpe[category]['mean'])
            dfs_sharpe[f'{category}_sharpe'] = dfs_sharpe[f'{category}_sharpe'].mask(np.isinf(dfs_sharpe[f'{category}_sharpe']), dfs_sharpe[category]['mean'])

        dfs_sharpe['Total_sharpe'] = dfs_sharpe['W_sharpe'] + 5 * dfs_sharpe['SVHLD_sharpe'] + dfs_sharpe['SO_sharpe'] - dfs_sharpe['WHIP_sharpe'] - dfs_sharpe['ERA_sharpe']

        pd.set_option('display.max_rows', None)

        print('Top Players By Risk Adjusted Wins')
        print(dfs_sharpe.sort_values(by = 'W_sharpe', ascending = False).head(10))
        
        print('Top Players By Risk Adjusted SVHLD')
        print(dfs_sharpe.sort_values(by = 'SVHLD_sharpe', ascending = False).head(10))
        
        print('Top Players By Risk Adjusted Strikeouts')
        print(dfs_sharpe.sort_values(by = 'SO_sharpe', ascending = False).head(10))
        
        print('Top Players By Risk Adjusted WHIP')
        print(dfs_sharpe.sort_values(by = 'WHIP_sharpe', ascending = True).head(10))
        
        print('Top Players By Risk Adjusted ERA')
        print(dfs_sharpe.sort_values(by = 'ERA_sharpe', ascending = True).head(10))

        print('Top Risk Adjusted Players')
        print(dfs_sharpe.sort_values(by = 'Total_sharpe', ascending = False).head(240))

        return dfs_sharpe


if __name__ == '__main__':
    read_csvs_batters()
    read_csvs_pitchers()
    read_csvs_pitchers(starters = False)
