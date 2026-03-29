import pandas as pd
from unidecode import unidecode
import numpy as np
import time

def read_csvs_batters(include_h = True):
    paths = [
            'projections/hitters/atc.csv',
            'projections/hitters/bat.csv',
            'projections/hitters/bat_x.csv',
            'projections/hitters/depth_charts.csv',
            'projections/hitters/oopsy.csv',
            'projections/hitters/steamer.csv',
            'projections/hitters/zips.csv',
            'projections/hitters/zips_dc.csv'
            ]
    position_path = 'projections/fangraphs-auction-calculator.csv'

    dfs = pd.concat([pd.read_csv(path, delimiter = '\t') for path in paths], ignore_index = True)

    positions = pd.read_csv(position_path)
    positions['Name'] = positions['Name'].apply(str.strip)

    #dfs = pd.merge(dfs, positions[['Name', 'POS']], on = 'Name', how = 'left')
    
    dfs_sharpe = dfs[["Name", "HR", "R", "RBI", "SB", "OBP", "H", "PA", "BB"] if include_h else ["Name", "HR", "R", "RBI", "SB", "OBP"]].groupby("Name").agg(['mean', 'std']).reset_index()
    #dfs_sharpe['POS'] = dfs['POS']
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
    dfs_sharpe = dfs_sharpe.sort_values(by = 'Total_sharpe', ascending = False)
    dfs_sharpe.index = np.arange(1, len(dfs_sharpe) + 1)

    pd.set_option('display.max_rows', None)
    dfs_sharpe['Name'] = dfs_sharpe['Name'].apply(unidecode)
    return dfs_sharpe

def read_csvs_pitchers(starters = True, include_expanded_stats = True):
    paths = [
            'projections/pitchers/atc.csv',
            'projections/pitchers/bat.csv',
            'projections/pitchers/bat_x.csv',
            'projections/pitchers/depth_charts.csv',
            'projections/pitchers/oopsy.csv',
            'projections/pitchers/steamer.csv',
            'projections/pitchers/zips.csv',
            'projections/pitchers/zips_dc.csv'
            ]

    if starters:
        dfs = pd.concat([pd.read_csv(path, delimiter = '\t') for path in paths], ignore_index = True)
        dfs = dfs[dfs['GS'] > 5]

        dfs_sharpe = dfs[["Name", "W", "SO", "WHIP", "ERA", 'ER', 'BB', 'H', 'IP'] if include_expanded_stats else ["Name", "W", "SO", "WHIP", "ERA"]].groupby("Name").agg(['mean', 'std']).reset_index()
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
        
        for category in ['W', 'SO']:
            dfs_sharpe[f'{category}_sharpe'] = (dfs_sharpe[category]['mean'] - replacement_player[category])/(dfs_sharpe[category]['std'])
            dfs_sharpe[f'{category}_sharpe'].fillna(dfs_sharpe[category]['mean'])
            dfs_sharpe[f'{category}_sharpe'] = dfs_sharpe[f'{category}_sharpe'].mask(np.isinf(dfs_sharpe[f'{category}_sharpe']), dfs_sharpe[category]['mean'])

        for category in ['WHIP', 'ERA']:
            dfs_sharpe[f'{category}_sharpe'] = (dfs_sharpe[category]['mean'] - replacement_player[category])/(dfs_sharpe[category]['std'])
            dfs_sharpe[f'{category}_sharpe'].fillna(dfs_sharpe[category]['mean'])
            dfs_sharpe[f'{category}_sharpe'] = dfs_sharpe[f'{category}_sharpe'].mask(np.isinf(dfs_sharpe[f'{category}_sharpe']), dfs_sharpe[category]['mean'])
            dfs_sharpe[f'{category}_sharpe'] *= -1


        dfs_sharpe['Total_sharpe'] = dfs_sharpe['W_sharpe'] + dfs_sharpe['SO_sharpe'] + dfs_sharpe['WHIP_sharpe'] + dfs_sharpe['ERA_sharpe']
        dfs_sharpe = dfs_sharpe.sort_values(by = 'Total_sharpe', ascending = False)
        dfs_sharpe.index = np.arange(1, len(dfs_sharpe) + 1)

        pd.set_option('display.max_rows', None)

        dfs_sharpe['Name'] = dfs_sharpe['Name'].apply(unidecode)
        return dfs_sharpe
    else:
        dfs = pd.concat([pd.read_csv(path, delimiter = '\t') for path in paths], ignore_index = True)
        dfs = dfs[dfs['G'] - dfs['GS'] > 5]
        dfs["SVHLD"] = dfs['SV'] + dfs['HLD']

        dfs_sharpe = dfs[["Name", "W", "SO", "SVHLD", "WHIP", "ERA", 'ER', 'BB', 'H', 'IP'] if include_expanded_stats else ["Name", "W", "SO", "SVHLD", "WHIP", "ERA"]].groupby("Name").agg(['mean', 'std']).reset_index()
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

        dfs_sharpe['WHIP_sharpe'] *= -1
        dfs_sharpe['ERA_sharpe'] *= -1

        dfs_sharpe['Total_sharpe'] = dfs_sharpe['W_sharpe'] + 5 * dfs_sharpe['SVHLD_sharpe'] + dfs_sharpe['SO_sharpe'] + dfs_sharpe['WHIP_sharpe'] + dfs_sharpe['ERA_sharpe']
        dfs_sharpe = dfs_sharpe.sort_values(by = 'Total_sharpe', ascending = False)
        dfs_sharpe.index = np.arange(1, len(dfs_sharpe) + 1)


        pd.set_option('display.max_rows', None)

        dfs_sharpe['Name'] = dfs_sharpe['Name'].apply(unidecode)
        return dfs_sharpe


if __name__ == '__main__':
    read_csvs_batters(include_h = False).to_csv("./batters_sharpe.csv")
    read_csvs_pitchers(starters = False).to_csv("./relievers_sharpe.csv")
    read_csvs_pitchers().to_csv("./starters_sharpe.csv")
