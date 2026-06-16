import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

if not mt5.initialize():
    print("MT5 Init failed")
    exit()

days = 30
date_from = datetime.now() - timedelta(days=days)
date_to = datetime.now() + timedelta(days=1)

deals = mt5.history_deals_get(date_from, date_to)
if deals is None:
    print("No deals")
else:
    print(f"Total deals: {len(deals)}")
    if len(deals) > 0:
        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        df = df[df['position_id'] > 0]
        
        for pos_id, group in df.groupby('position_id'):
            ins = group[group['entry'] == 0]
            outs = group[group['entry'] == 1]
            if len(ins) > 0 and len(outs) > 0:
                print(f"Position {pos_id}: IN: {len(ins)} deals, OUT: {len(outs)} deals, PnL: {outs['profit'].sum()}")

mt5.shutdown()
