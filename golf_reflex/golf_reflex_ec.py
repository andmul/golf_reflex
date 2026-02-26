import reflex as rx
import reflex_echarts as rx_echarts
import pandas as pd
import numpy as np
import os
import ast
import json
from typing import List, Optional

MODULE_DIR: str = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR: str = os.path.join(os.path.dirname(MODULE_DIR), "assets")

# --- 1. DATA LOADING & ROBUST PRE-PROCESSING ---
def load_and_prep_data(dataset_name: str = "AK50"):
    file_path = os.path.join(ASSETS_DIR, f"{dataset_name}.parquet")
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        # Fallback to CSV if parquet not found and requesting AK50 (for migration safety)
        if dataset_name == "AK50":
            csv_path = os.path.join(ASSETS_DIR, "data.csv")
            if os.path.exists(csv_path):
                print(f"Parquet not found. Fallback to CSV: {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                print(f"Error: Data file not found at {file_path} or {csv_path}")
                return pd.DataFrame()
        else:
            print(f"Error: Data file not found at {file_path}")
            return pd.DataFrame()
    else:
        df = pd.read_parquet(file_path)

    df['Datum'] = pd.to_datetime(df['Datum'])
    df = df.sort_values('Datum')

    # Fill string NaNs
    df[['Turnier', 'Club', 'Spielmodus']] = df[['Turnier', 'Club', 'Spielmodus']].fillna("-")

    # Name processing
    df['Vorname'] = df['Vorname'].fillna('').astype(str).str.strip()
    df['Name'] = df['Name'].fillna('').astype(str).str.strip()

    mask_both = (df['Vorname'] != '') & (df['Name'] != '')
    mask_name_only = (df['Vorname'] == '') & (df['Name'] != '')
    mask_vorname_only = (df['Vorname'] != '') & (df['Name'] == '')

    df['Spieler_Name'] = 'Unknown'
    df.loc[mask_both, 'Spieler_Name'] = df.loc[mask_both, 'Name'] + ', ' + df.loc[mask_both, 'Vorname']
    df.loc[mask_name_only, 'Spieler_Name'] = df.loc[mask_name_only, 'Name']
    df.loc[mask_vorname_only, 'Spieler_Name'] = df.loc[mask_vorname_only, 'Vorname']

    # --- DEBUGGING VGW (HCPRelevant) ---
    if 'HCPRelevant' in df.columns:
        # Convert to string and clean
        df['HCPRelevant'] = df['HCPRelevant'].astype(str).str.lower().str.strip()

        # Expanded Map for German/English/Numbers
        true_values = ['true', 'yes', '1', '1.0', 'ja', 't', 'y', 'j']
        df['HCPRelevant'] = df['HCPRelevant'].isin(true_values)
    else:
        print("ERROR: 'HCPRelevant' column NOT found!")
        df['HCPRelevant'] = False

    # --- STATS CALCULATION ---
    # Parquet files might have 'par' as a list/array already, CSV as string
    if 'par' in df.columns and 'score' in df.columns:
        def calc_row_deltas(row):
            try:
                # Handle par list
                if isinstance(row['par'], str):
                    p_list = ast.literal_eval(row['par'])
                elif isinstance(row['par'], (list, np.ndarray)):
                    p_list = list(row['par'])
                else:
                    p_list = []

                # Handle score list
                if isinstance(row['score'], str):
                    s_list = ast.literal_eval(row['score'])
                elif isinstance(row['score'], (list, np.ndarray)):
                    s_list = list(row['score'])
                else:
                    s_list = []

                if not p_list or not s_list: return [0, 0, 0, 0]

                birdies, eagles, albatross, aces = 0, 0, 0, 0

                front_nine = list(range(0, 9))
                back_nine = list(range(10, 19))
                valid_indices = front_nine + back_nine

                for i in valid_indices:
                    if i >= len(p_list) or i >= len(s_list): continue
                    par_val = p_list[i]
                    score_val = s_list[i]
                    if (pd.isna(par_val) or pd.isna(score_val) or par_val == 0 or score_val == 0): continue

                    diff = score_val - par_val
                    if score_val == 1: aces += 1
                    if diff == -1: birdies += 1
                    elif diff == -2 and score_val != 1: eagles += 1
                    elif diff == -3 and score_val != 1: albatross += 1

                return [birdies, eagles, albatross, aces]
            except:
                return [0,0,0,0]

        stats_data = df.apply(calc_row_deltas, axis=1, result_type='expand')
        df[['cnt_birdie', 'cnt_eagle', 'cnt_albatross', 'cnt_ace']] = stats_data

    df['Jahr'] = df['Datum'].dt.year
    return df


# Helper for sorting
def get_sort_key(name):
    if ',' in name: return name.split(',')[0].strip().lower()
    return name.lower()

# Secure password handling: Use env var, fallback to default if not set (for backward compat/demo)
APP_PASSWORD = os.environ.get("APP_PASSWORD", "nobi")


class GolfStateEc(rx.State):
    is_authenticated: bool = False
    password_input: str = ""
    auth_error: str = ""

    # Filter states
    filter_brutto: bool = False
    filter_turnierart: str = "Alle"
    selected_player: str = "Alle Spieler"
    selected_year: str = "2025"

    # Dataset Selection
    selected_dataset: str = "AK50"
    _data: pd.DataFrame = pd.DataFrame()

    def on_load(self):
        """Called when the app loads. Initializes default dataset."""
        self._reload_data()

    def _reload_data(self):
        self._data = load_and_prep_data(self.selected_dataset)
        # Reset filters if needed, or validate them against new data
        if "Spieler_Name" in self._data.columns:
            # Check if current selected player is valid, else reset
            players = self.player_options
            if self.selected_player not in players:
                 self.selected_player = "Alle Spieler"
        if "Jahr" in self._data.columns:
             years = self.year_options
             if self.selected_year not in years:
                 # Default to most recent year if available
                 if len(years) > 1:
                     self.selected_year = years[1] # [0] is "Alle Jahre"
                 else:
                     self.selected_year = "Alle Jahre"


    def set_selected_dataset(self, value: str):
        self.selected_dataset = value
        self._reload_data()

    @rx.var
    def available_datasets(self) -> list[str]:
        files = [f for f in os.listdir(ASSETS_DIR) if f.endswith('.parquet')]
        return sorted([os.path.splitext(f)[0] for f in files])

    @rx.var
    def player_options(self) -> list[str]:
        if self._data.empty: return ["Alle Spieler"]
        unique_players = sorted(self._data['Spieler_Name'].unique().tolist(), key=get_sort_key)
        return ["Alle Spieler"] + unique_players

    @rx.var
    def year_options(self) -> list[str]:
        if self._data.empty: return ["Alle Jahre"]
        unique_years = sorted(self._data['Jahr'].unique().tolist(), reverse=True)
        return ["Alle Jahre"] + [str(year) for year in unique_years]

    # --- AUTH ---
    def set_password_input(self, value: str):
        self.password_input = value

    def check_password(self):
        if self.password_input == APP_PASSWORD:
            self.is_authenticated = True
            self.auth_error = ""
            self.password_input = ""
            # Ensure data is loaded
            if self._data.empty:
                self._reload_data()
        else:
            self.auth_error = "Falsches Passwort"

    def handle_key_down(self, key: str):
        if key == "Enter": return self.check_password()

    def logout(self):
        self.is_authenticated = False
        self.auth_error = ""
        self.password_input = ""

    # --- FILTERS ---
    def toggle_brutto(self, value: bool):
        self.filter_brutto = value

    def set_filter_turnierart(self, value: str):
        self.filter_turnierart = value

    def set_selected_player(self, value: str):
        self.selected_player = value

    def set_selected_year(self, value: str):
        self.selected_year = value

    # --- DATA LAYERS ---

    @rx.var
    def period_df(self) -> pd.DataFrame:
        """
        Base Data for the selected Period (Player + Year).
        Independent of Checkboxes (Brutto/Einzel/VGW).
        Used for the 'Turnier Übersicht' cards.
        """
        if not self.is_authenticated or self._data.empty: return pd.DataFrame()

        df = self._data.copy()

        if self.selected_player != "Alle Spieler":
            df = df[df['Spieler_Name'] == self.selected_player]

        if self.selected_year != "Alle Jahre":
            df = df[df['Jahr'] == int(self.selected_year)]

        return df

    @rx.var
    def filtered_df(self) -> pd.DataFrame:
        """
        Final Data for Charts & Lists.
        Applies Checkboxes ON TOP of period_df.
        """
        df = self.period_df.copy()
        if df.empty: return df

        # 1. Nur Turniere mit Ergebnis
        if self.filter_brutto:
            # Check for non-zero and non-NaN
            df = df[(df['Brutto'].notna()) & (df['Brutto'] != 0)]

        # 2. Turnierart Filter
        if self.filter_turnierart == "Einzel":
            # "Einzel" filter: Includes all tournaments where Spielmodus starts with "Einzel".
            # This excludes team formats like Scramble, Vierer, etc.
            df = df[df['Spielmodus'].str.startswith('Einzel', na=False)]
        elif self.filter_turnierart == "Vorgabewirksam":
             # "Vorgabewirksam" filter: A subset of Einzel.
             # Includes only tournaments explicitly marked as HCPRelevant (VGW) in the database.
             df = df[df['HCPRelevant'] == True]

        return df.sort_values('Datum').reset_index(drop=True)

    @rx.var
    def tournament_counts(self) -> dict:
        """
        Counts based on period_df (ignoring checkboxes)
        to show the full potential of the selection.
        """
        df = self.period_df
        if df.empty:
            return {"total": 0, "with_result": 0, "single": 0, "vgw": 0}

        return {
            "total": len(df),
            "with_result": len(df[(df['Brutto'].notna()) & (df['Brutto'] != 0)]),
            "single": len(df[df['Spielmodus'].str.startswith('Einzel', na=False)]),
            "vgw": len(df[df['HCPRelevant'] == True])
        }

    @rx.var
    def stats_title(self) -> str:
        p = "alle Spieler" if self.selected_player == "Alle Spieler" else self.selected_player
        y = "allen Jahren" if self.selected_year == "Alle Jahre" else self.selected_year
        return f"Für {p} im Jahr {y}"

    @rx.var
    def scoring_stats(self) -> dict:
        # Scoring stats depend on the filtered view (what you see in the chart)
        if self.filtered_df.empty or 'cnt_birdie' not in self.filtered_df.columns:
            return {"birdie": 0, "eagle": 0, "albatross": 0, "ace": 0}

        df = self.filtered_df
        return {
            "birdie": int(df['cnt_birdie'].sum()),
            "eagle": int(df['cnt_eagle'].sum()),
            "albatross": int(df['cnt_albatross'].sum()),
            "ace": int(df['cnt_ace'].sum()),
        }

    @rx.var
    def scoring_total(self) -> int:
        s = self.scoring_stats
        return s["birdie"] + s["eagle"] + s["albatross"] + s["ace"]

    @rx.var
    def scoring_average(self) -> str:
        total = self.scoring_total
        count = len(self.filtered_df)
        if count > 0:
            return f"{total / count:.1f}"
        return "0.0"

    @rx.var
    def last_five_rounds(self) -> list[dict]:
        df = self.filtered_df
        if df.empty: return []

        recent = df.tail(10).iloc[::-1].copy()

        rounds = []
        for _, r in recent.iterrows():
            upar_val = ""
            if not pd.isna(r.get('uPar')):
                upar_val = f"{int(r['uPar']):+d}"

            rounds.append({
                "date": r["Datum"].strftime('%d.%m.%y'),
                "club": str(r["Club"]),
                "turnier": str(r["Turnier"]),
                "hcp": f"HCP: {r['HCP']}",
                "upar": upar_val,
                "brutto": f"Brutto: {int(r['Brutto'])}",
                "player": str(r["Spieler_Name"])
            })
        return rounds

    @rx.var
    def player_summary_list(self) -> list[dict]:
        if self.selected_player != "Alle Spieler" or self.filtered_df.empty:
            return []
        df = self.filtered_df
        summary = df.groupby('Spieler_Name').agg(
            Rounds=('Turnier', 'count'),
            Avg_Brutto=('Brutto', 'mean'),
            Last_HCP=('HCP', 'last')
        ).reset_index()
        summary['Avg_Brutto'] = summary['Avg_Brutto'].round(1)
        return summary.to_dict('records')

    @rx.var
    def echarts_option(self) -> dict:
        if self.filtered_df.empty:
            return {}

        df = self.filtered_df.copy()
        df['ContinuousIndex'] = range(len(df))

        dates = df['Datum'].dt.strftime('%d.%m.%y').tolist()
        brutto_data = df['Brutto'].astype(int).tolist()
        hcp_data = df['HCP'].tolist()

        # Determine uPar data for labels
        upar_data = []
        for _, row in df.iterrows():
            if not pd.isna(row.get('uPar')):
                val = int(row['uPar'])
                upar_data.append({"value": row['Brutto'], "symbol": "none", "label": {"show": True, "formatter": f"{val:+}", "position": "top", "color": "black", "fontSize": 9, "fontWeight": "bold"}})
            else:
                upar_data.append(None) # Or appropriate empty value

        # Winter breaks lines
        mark_lines = []
        if len(df) > 1:
            date_diffs = df['Datum'].diff().dt.days
            gap_indices = df.index[date_diffs > 50].tolist()
            for idx in gap_indices:
                if idx > 0:
                    try:
                         # Calculate position similar to Plotly logic. ECharts xAxis is categorical by default (index based)
                         # so we can use the index directly.
                         # Since ContinuousIndex is 0..N-1, we can find the split point.
                         # Plotly was: (pos_current + pos_prev) / 2
                         # Here: we need the index in the current filtered df

                         # Get integer location in the current df
                         iloc_idx = df.index.get_loc(idx)
                         # The gap is before this index, so between iloc_idx-1 and iloc_idx
                         mark_pos = iloc_idx - 0.5
                         mark_lines.append({"xAxis": mark_pos})
                    except:
                        pass

        series = []

        # Brutto Bar Series
        series.append({
            "name": "Brutto",
            "type": "bar",
            "data": brutto_data,
            "yAxisIndex": 1,
            "itemStyle": {"color": "rgba(84, 245, 66, 0.6)"},
            "label": {
                "show": True,
                "position": "inside",
                "color": "black",
                "fontSize": 14
            },
            "tooltip": {"show": False} # We use a shared tooltip
        })

        # HCP Line Series
        if self.selected_player != "Alle Spieler":
            series.append({
                "name": "HCP",
                "type": "line",
                "data": hcp_data,
                "yAxisIndex": 0,
                "smooth": False,
                "symbol": "circle",
                "symbolSize": 6,
                "itemStyle": {"color": "royalblue"},
                "lineStyle": {"width": 1.5, "color": "royalblue"},
                "tooltip": {"show": False}
            })

        # uPar Scatter/Label Series (hacky way to put labels on top if we want separate series,
        # but ECharts supports rich labels. Alternatively, use a scatter series on top of bars)
        # The Plotly code used a separate scatter trace for text.
        # In ECharts, we can try adding a second scatter series with transparency just for labels.

        # Filter valid uPar points
        upar_points = []
        for i, val in enumerate(upar_data):
            if val is not None:
                upar_points.append([i, val["value"], val["label"]["formatter"]])

        if upar_points:
             series.append({
                "name": "Über Par",
                "type": "scatter",
                "data": [[p[0], p[1]] for p in upar_points],
                "yAxisIndex": 1,
                "symbolSize": 1, # Tiny symbol
                "itemStyle": {"opacity": 0}, # Invisible
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{b}", # We need to pass the label text.
                    # Actually better to use 'data' as objects with label prop?
                    # Let's reconstruct data
                },
                 # Re-doing data construction for this series
                "data": [
                    {
                        "value": [p[0], p[1]],
                        "label": {
                            "show": True,
                            "formatter": p[2],
                            "position": "top",
                            "color": "black",
                            "fontSize": 10,
                            "fontWeight": "bold",
                            "distance": 5
                        }
                    } for p in upar_points
                ],
                "z": 10,
                "tooltip": {"show": False}
             })

        # Prepare tooltip data mapping
        # We need to access row data in the tooltip formatter.
        # ECharts formatter callbacks run in JS. passing massive data might be tricky.
        # Standard approach: Put all info into 'data' or mapped arrays and access via index.
        # However, Reflex runs Python. We can construct a list of tooltip strings and access them by index.

        tooltip_texts = df.apply(lambda r: (
            f"<div style='text-align:left;'>"
            f"<b>{r['Datum'].strftime('%d.%m.%y')}</b><br>"
            f"{r['Turnier']}<br>"
            f"{r['Club']}<br>"
            f"<b>Spieler:</b> {r['Spieler_Name']}<br>"
            f"<b>Spielmodus:</b> {r.get('Spielmodus', 'N/A')}<br>"
            f"<b>Brutto:</b> {int(r['Brutto'])}<br>"
            f"<b>Par:</b> {f'{int(r.get('Par'))} | {int(r.get('uPar', 0)):+,d}' if not pd.isna(r.get('uPar', None)) else r.get('Par')}<br>"
            f"<b>CR:</b> {int(r['cr']) if pd.notna(r.get('cr')) else 'N/A'}<br>"
            f"</div>"
        ) if pd.notna(r.get('Par')) else "", axis=1).tolist()

        # Safe handling if Par is nan, just in case, though the above logic tries to handle it.
        # Better: Ensure all required fields have defaults before generating string.

        # Serialize to JSON string to be safely embedded in the JS function
        tooltip_json = json.dumps(tooltip_texts)

        option = {
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "backgroundColor": "rgba(255, 255, 255, 0.9)",
                "padding": 10,
                "textStyle": {"color": "#333"},
                "extraCssText": "box-shadow: 0 0 3px rgba(0, 0, 0, 0.3);",
                 # JS function string to render custom html.
                 # params[0].dataIndex gives the index.
                 # We need to inject the tooltip_texts array into the JS context.
                "formatter": f"""function (params) {{
                    var texts = {tooltip_json};
                    var index = params[0].dataIndex;
                    return texts[index];
                }}"""
            },
            "legend": {
                "data": ["HCP", "Brutto"],
                "top": 0
            },
            "grid": {
                "left": "3%",
                "right": "3%",
                "bottom": "15%", # Space for dataZoom
                "containLabel": True
            },
            "xAxis": {
                "type": "category",
                "data": dates,
                "axisTick": {"alignWithLabel": True},
                "axisLine": {"lineStyle": {"color": "#ccc"}},
                "axisLabel": {"color": "#333"}
            },
            "yAxis": [
                {
                    "type": "value",
                    "name": "HCP",
                    "position": "left",
                    "axisLine": {"show": True, "lineStyle": {"color": "royalblue"}},
                    "axisLabel": {"formatter": "{value}"},
                    "splitLine": {"show": True, "lineStyle": {"type": "dashed"}}
                },
                {
                    "type": "value",
                    "name": "Brutto",
                    "position": "right",
                    "axisLine": {"show": True, "lineStyle": {"color": "green"}},
                    "axisLabel": {"formatter": "{value}"},
                    "splitLine": {"show": False}
                }
            ],
            "dataZoom": [
                {
                    "type": "slider",
                    "show": True,
                    "xAxisIndex": [0],
                    "start": 0, # Should calculate based on last 20 items or similar logic
                    "end": 100,
                    "bottom": 10
                },
                {
                    "type": "inside",
                    "xAxisIndex": [0],
                    "start": 0,
                    "end": 100
                }
            ],
            "series": series
        }

        # Handle initial zoom logic (show last 20 items roughly)
        total_bars = len(df)
        if total_bars > 20:
             start_pct = max(0, (total_bars - 20) / total_bars * 100)
             option["dataZoom"][0]["start"] = start_pct
             option["dataZoom"][1]["start"] = start_pct

        # Add MarkLines for Winter Breaks
        if mark_lines:
             # Add to the Brutto series (index 0 if HCP hidden, or index 1)
             # Better to add to the first series present.
             if series:
                 series[0]["markLine"] = {
                     "symbol": "none",
                     "label": {"show": False},
                     "lineStyle": {"type": "dashed", "color": "red", "opacity": 0.4},
                     "data": mark_lines
                 }

        return option


# --- UI COMPONENTS ---

def round_card(r: dict):
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text(r["date"], font_weight="bold", font_size="0.8em"),
                    rx.text(r["club"], font_size="0.7em", color="gray"),
                    rx.cond(
                        GolfStateEc.selected_player == "Alle Spieler",
                        rx.text(r["player"], font_size="0.7em", color="#2c5282", font_weight="bold"),
                    ),
                    align_items="start", spacing="0"
                ),
                rx.spacer(),
                rx.text(r["turnier"], font_size="0.65em", font_style="italic", color="gray", max_width="200px",
                        text_align="center"),
                rx.spacer(),
                rx.vstack(
                    rx.hstack(
                        rx.text(r["hcp"], font_size="0.7em", color="royalblue"),
                        rx.text(r["upar"], font_size="0.75em", font_weight="bold", color="blue"),
                        spacing="2"
                    ),
                    rx.text(r["brutto"], font_weight="bold", color="green", font_size="0.75em"),
                    align_items="end", spacing="0"
                ),
                width="100%", align_items="center"
            ),
            border_bottom="1px solid #EDEDED", padding_y="0.5em", width="100%"
        ),
        width="100%"
    )


def player_summary_row(row: dict):
    return rx.table.row(
        rx.table.cell(row["Spieler_Name"]),
        rx.table.cell(row["Rounds"]),
        rx.table.cell(row["Avg_Brutto"]),
        rx.table.cell(row["Last_HCP"]),
    )


def player_summary_table():
    return rx.box(
        rx.vstack(
            rx.heading("Spieler Übersicht", size="3", padding_bottom="0.5em"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Spieler"),
                        rx.table.column_header_cell("Runden"),
                        rx.table.column_header_cell("Ø Brutto"),
                        rx.table.column_header_cell("Akt. HCP"),
                    )
                ),
                rx.table.body(rx.foreach(GolfStateEc.player_summary_list, player_summary_row)),
                width="100%", variant="surface"
            ),
        ),
        width="100%", padding="1em", background_color="#fcfcfc",
        border_radius="lg", border="1px solid #e0e0e0", margin_top="1em"
    )


def scoring_stats_card():
    def stat_box(label, value, color, bg_color):
        return rx.vstack(
            rx.text(label, font_size="0.8em", color="gray"),
            rx.text(value, font_size="1.8em", font_weight="bold", color=color),
            align="center", spacing="0", padding="0.8em", border_radius="md",
            background_color=bg_color, width="100%", border=f"1px solid {color}4D"
        )

    return rx.box(
        rx.vstack(
            rx.heading("Scoring Statistiken", size="3"),
            rx.text(GolfStateEc.stats_title, font_size="0.8em", color="gray"),
            rx.hstack(
                stat_box("Birdies", GolfStateEc.scoring_stats["birdie"], "#FFD700", "rgba(255, 215, 0, 0.1)"),
                stat_box("Eagles", GolfStateEc.scoring_stats["eagle"], "#FF8C00", "rgba(255, 140, 0, 0.1)"),
                stat_box("Albatros", GolfStateEc.scoring_stats["albatross"], "#8B00FF", "rgba(139, 0, 255, 0.1)"),
                stat_box("Ace", GolfStateEc.scoring_stats["ace"], "#FF0000", "rgba(255, 0, 0, 0.1)"),
                width="100%", spacing="2", justify="between"
            ),
            rx.hstack(
                rx.text("Gesamt: ", font_size="0.9em", color="gray"),
                rx.text(GolfStateEc.scoring_total, font_size="0.9em", font_weight="bold"),
                rx.spacer(),
                rx.text("Ø pro Runde: ", font_size="0.9em", color="gray"),
                rx.text(GolfStateEc.scoring_average, font_size="0.9em", font_weight="bold"),
                width="100%", justify="between", padding_top="0.5em"
            ),
            width="100%", padding="1em", background_color="#fcfcfc", border_radius="lg", border="1px solid #e0e0e0",
            box_shadow="sm"
        ),
        width="100%"
    )


def tournament_summary_card():
    return rx.box(
        rx.vstack(
            rx.heading("Turnier Übersicht", size="3"),
            rx.grid(
                rx.vstack(
                    rx.text("Gesamt", font_size="0.8em", color="gray"),
                    rx.text(GolfStateEc.tournament_counts["total"], font_size="1.5em", font_weight="bold"),
                    align="center", spacing="0", padding="0.5em", border_radius="md", background_color="#f0f0f0",
                    width="100%"
                ),
                rx.vstack(
                    rx.text("Mit Ergebnis", font_size="0.8em", color="gray"),
                    rx.text(GolfStateEc.tournament_counts["with_result"], font_size="1.5em", font_weight="bold",
                            color="green"),
                    align="center", spacing="0", padding="0.5em", border_radius="md",
                    background_color="rgba(0, 255, 0, 0.05)", width="100%"
                ),
                rx.vstack(
                    rx.text("Einzel", font_size="0.8em", color="gray"),
                    rx.text(GolfStateEc.tournament_counts["single"], font_size="1.5em", font_weight="bold", color="blue"),
                    align="center", spacing="0", padding="0.5em", border_radius="md",
                    background_color="rgba(0, 0, 255, 0.05)", width="100%"
                ),
                rx.vstack(
                    rx.text("Vorgabewirksam", font_size="0.8em", color="gray"),
                    rx.text(GolfStateEc.tournament_counts["vgw"], font_size="1.5em", font_weight="bold", color="purple"),
                    align="center", spacing="0", padding="0.5em", border_radius="md",
                    background_color="rgba(128, 0, 128, 0.05)", width="100%"
                ),
                columns="2", spacing="2", width="100%",  # Changed to 2 columns for better layout with 4 items
            ),
            width="100%", padding="1em", background_color="#fcfcfc", border_radius="lg", border="1px solid #e0e0e0",
        ),
        width="100%"
    )


def login_form():
    return rx.center(
        rx.box(
            rx.vstack(
                rx.heading("Golf Performance Dashboard", size="6", padding_bottom="1em"),
                rx.text("Bitte Passwort eingeben", color="gray", padding_bottom="0.5em"),
                rx.input(
                    value=GolfStateEc.password_input,
                    on_change=GolfStateEc.set_password_input,
                    on_key_down=GolfStateEc.handle_key_down,
                    placeholder="Passwort",
                    type="password",
                    width="250px",
                    aria_label="Passwort Eingabefeld",
                ),
                rx.cond(
                    GolfStateEc.auth_error,
                    rx.text(GolfStateEc.auth_error, color="red", font_size="0.9em"),
                ),
                rx.button(
                    "Anmelden",
                    on_click=GolfStateEc.check_password,
                    width="250px",
                    margin_top="1em",
                ),
                spacing="1", align="center",
            ),
            padding="3em", border="1px solid #e0e0e0", border_radius="lg",
            background_color="white", box_shadow="lg",
        ),
        width="100vw", height="100vh",
        background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    )


def dashboard():
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("🏌️ Golf Performance", size="6"),
                rx.spacer(),
                rx.button("Abmelden", on_click=GolfStateEc.logout, size="2", variant="soft"),
                width="100%", padding_y="0.4em",
            ),

            # --- DATASET SELECTOR ---
            rx.hstack(
                rx.text("Datensatz:", font_size="0.9em", font_weight="bold"),
                rx.select(
                    GolfStateEc.available_datasets,
                    value=GolfStateEc.selected_dataset,
                    on_change=GolfStateEc.set_selected_dataset,
                    width="100%", max_width="400px",
                    custom_attrs={"aria-label": "Datensatz Auswahl"}
                ),
                width="100%", padding_bottom="1em", align="center", spacing="2"
            ),

            rx.vstack(
                rx.hstack(
                    rx.text("Spieler:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(GolfStateEc.player_options, value=GolfStateEc.selected_player,
                              on_change=GolfStateEc.set_selected_player, width="100%", max_width="400px",
                              custom_attrs={"aria-label": "Spieler Auswahl"}),
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Jahr:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(GolfStateEc.year_options, value=GolfStateEc.selected_year,
                              on_change=GolfStateEc.set_selected_year, width="100%", max_width="400px",
                              custom_attrs={"aria-label": "Jahr Auswahl"}),
                    width="100%",
                ),
                width="100%", spacing="2", padding_bottom="1em",
            ),

            # --- UPDATED TOGGLES ---
            rx.hstack(
                rx.hstack(
                    rx.text("Turnierart:", font_size="0.9em", font_weight="bold"),
                    rx.select(
                        ["Alle", "Einzel", "Vorgabewirksam"],
                        value=GolfStateEc.filter_turnierart,
                        on_change=GolfStateEc.set_filter_turnierart,
                        width="180px",
                        custom_attrs={"aria-label": "Turnierart Auswahl"}
                    ),
                    align="center", spacing="2"
                ),
                rx.hstack(rx.checkbox(checked=GolfStateEc.filter_brutto, on_change=GolfStateEc.toggle_brutto),
                          rx.text("nur Turniere mit Ergebnis", font_size="0.9em"), align="center", spacing="1"),
                justify="center", width="100%", spacing="4", padding_bottom="1em", wrap="wrap"
            ),

            rx.box(
                rx.cond(
                    GolfStateEc.filtered_df.empty,
                    rx.text("Keine Daten", color="gray", padding="40px", text_align="center"),
                    rx_echarts.echarts(
                        option=GolfStateEc.echarts_option,
                        style={"width": "100%", "height": "100%"},
                    )
                ),
                width="100%", min_height="450px", height="55vh", border_radius="md", border="1px solid #e0e0e0",
                background_color="white"
            ),

            scoring_stats_card(),
            tournament_summary_card(),

            rx.vstack(
                rx.heading("Letzte 10 Runden", size="3"),
                rx.foreach(GolfStateEc.last_five_rounds, round_card),
                width="100%", background_color="#fcfcfc", padding="1.2em", border_radius="lg",
                border="1px solid #e0e0e0", box_shadow="sm"
            ),

            # --- CONDITIONAL TABLE ---
            rx.cond(
                GolfStateEc.selected_player == "Alle Spieler",
                player_summary_table(),
            ),

            width="100%", spacing="3",
        ),
        padding="1em", max_width="800px", margin="0 auto",
    )


def index():
    return rx.cond(GolfStateEc.is_authenticated, dashboard(), login_form())


app = rx.App()
app.add_page(index, on_load=GolfStateEc.on_load)
