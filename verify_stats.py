import sys
from unittest.mock import MagicMock

# Mock reflex and its state
mock_rx = MagicMock()
mock_state = MagicMock()
mock_rx.State = MagicMock
sys.modules['reflex'] = mock_rx

# Mock GolfState specifically to have on_load
class MockGolfState:
    on_load = MagicMock()

# We need to inject this into the module namespace before importing
# However, since the class is defined IN the module, this is tricky.
# Instead, we'll let the module define the class, but we need to prevent the app.add_page call from failing.
# Or better, just import load_and_prep_data directly if possible, but the module executes on import.

# Let's mock app.add_page to do nothing
mock_rx.App.return_value.add_page = MagicMock()

from golf_reflex.golf_reflex import load_and_prep_data

print("Loading data...")
df = load_and_prep_data("AK50")
if df.empty:
    print("DataFrame is empty!")
else:
    print(f"Rows loaded: {len(df)}")
    if 'cnt_birdie' in df.columns:
        print(f"Birdies: {df['cnt_birdie'].sum()}")
        print(f"Eagles: {df['cnt_eagle'].sum()}")
        print(f"Albatross: {df['cnt_albatross'].sum()}")
        print(f"Aces: {df['cnt_ace'].sum()}")
    else:
        print("Stats columns missing!")
