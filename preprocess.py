import os
from astropy.io import fits
import pandas as pd
import numpy as np

base_path = "data/"

def process_lc_and_gti(lc_path):
    gti_path = lc_path.replace('.lc', '.gti')
    try:
        with fits.open(lc_path) as hdul:
            data = hdul[1].data
            time = np.array(data['TIME']).byteswap().newbyteorder()
            flux = np.array(data['COUNTS']).byteswap().newbyteorder()
        df = pd.DataFrame({'TIME': time, 'FLUX': flux})

        with fits.open(gti_path) as gti:
            gti_data = gti[1].data
            start = np.array(gti_data['START']).byteswap().newbyteorder()
            stop = np.array(gti_data['STOP']).byteswap().newbyteorder()
            gti_ranges = list(zip(start, stop))

        def is_in_gti(t): return any(s <= t <= e for s, e in gti_ranges)
        df['IS_VALID'] = df['TIME'].apply(is_in_gti)
        df = df[df['IS_VALID']].drop(columns='IS_VALID')
        df['DATETIME'] = pd.to_datetime(df['TIME'], unit='s', origin='unix')
        return df

    except Exception as e:
        print(f"Error processing {lc_path}: {e}")
        return None

all_dfs = []
for root, _, files in os.walk(base_path):
    if "SDD2" in root:
        for file in files:
            if file.endswith(".lc"):
                full_path = os.path.join(root, file)
                df = process_lc_and_gti(full_path)
                if df is not None:
                    all_dfs.append(df)

if all_dfs:
    final_df = pd.concat(all_dfs).sort_values(by="DATETIME").reset_index(drop=True)

    # Smoothing: 1-min rolling window centered
    final_df['FLUX_SMOOTH'] = final_df['FLUX'].rolling(window=60, center=True, min_periods=1).mean()

    # Flare detection: flux above (median + 3×std)
    threshold = final_df['FLUX_SMOOTH'].median() + 3 * final_df['FLUX_SMOOTH'].std()
    final_df['FLARE'] = (final_df['FLUX_SMOOTH'] > threshold).astype(int)

    # Resample to 5-minute intervals
    resampled = final_df.set_index('DATETIME').resample('5min').mean().dropna().reset_index()

    # Target: flare in next 24h (288 5-min bins)
    resampled['FLARE_TARGET_24h'] = (
        resampled['FLARE']
        .rolling(window=288, min_periods=1)
        .max()
        .shift(-288)
        .fillna(0)
        .astype(int)
    )

    output_file = os.path.join(base_path, "processed_solexs_tft_ready.csv")
    resampled.to_csv(output_file, index=False)
    print(f"Final dataset with labels saved to: {output_file}")

else:
    print("No .lc data processed.")
