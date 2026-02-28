import reflex as rx
import reflex_echarts as rx_echarts
import pandas as pd
import numpy as np
import os
import ast

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
        df['HCPRelevant'] = False

    # --- STATS CALCULATION ---
    # Parquet files might have 'par' as a list/array already, CSV as string
    if 'par' in df.columns and 'score' in df.columns:
        def calc_row_deltas(row):
            try:
                if isinstance(row['par'], str):
                    p_list = ast.literal_eval(row['par'])
                elif isinstance(row['par'], (list, np.ndarray)):
                    p_list = list(row['par'])
                else:
                    p_list = []

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


class GolfState(rx.State):
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
        if "Spieler_Name" in self._data.columns:
            players = self.player_options
            if self.selected_player not in players:
                 self.selected_player = "Alle Spieler"
        if "Jahr" in self._data.columns:
             years = self.year_options
             if self.selected_year not in years:
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
        if not self.is_authenticated or self._data.empty: return pd.DataFrame()
        df = self._data.copy()
        if self.selected_player != "Alle Spieler":
            df = df[df['Spieler_Name'] == self.selected_player]
        if self.selected_year != "Alle Jahre":
            df = df[df['Jahr'] == int(self.selected_year)]
        return df

    @rx.var
    def filtered_df(self) -> pd.DataFrame:
        df = self.period_df.copy()
        if df.empty: return df

        if self.filter_brutto:
            df = df[(df['Brutto'].notna()) & (df['Brutto'] != 0)]

        if self.filter_turnierart == "Einzel":
            df = df[df['Spielmodus'].str.startswith('Einzel', na=False)]
        elif self.filter_turnierart == "Vorgabewirksam":
             df = df[df['HCPRelevant'] == True]

        return df.sort_values('Datum').reset_index(drop=True)

    @rx.var
    def tournament_counts(self) -> dict:
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
        """
        Generates the ECharts configuration.
        Optimized to pass raw data via `dataset` and use pure JS formatters
        to avoid frontend serialization/dispatch errors and ensure performance.
        """
        if self.filtered_df.empty:
            return {}

        df = self.filtered_df.copy()
        # Create a continuous index for the x-axis to evenly space bars despite date gaps
        df['ContinuousIndex'] = range(len(df))

        # 1. Prepare Dates for the X-Axis Category
        dates = df['Datum'].dt.strftime('%d.%m.%y').tolist()

        # 2. Prepare Data for uPar Scatter Labels
        # We use a separate data array mapped to the continuous index so we can precisely
        # position the labels at the top of the bars.
        upar_data = []
        for i, row in df.iterrows():
            if not pd.isna(row.get('uPar')):
                val = int(row['uPar'])
                # Format: [X-Index (integer), Y-Value (Brutto), Label Text]
                upar_data.append([row['ContinuousIndex'], row['Brutto'], f"{val:+}"])

        # 3. Calculate Winter Breaks (Gaps > 50 days)
        mark_lines = []
        if len(df) > 1:
            date_diffs = df['Datum'].diff().dt.days
            gap_indices = df.index[date_diffs > 50].tolist()
            for idx in gap_indices:
                if idx > 0:
                    try:
                         iloc_idx = df.index.get_loc(idx)
                         # Place the line exactly between the two bars
                         mark_pos = iloc_idx - 0.5
                         mark_lines.append({"xAxis": mark_pos})
                    except:
                        pass

        # 4. Construct the Main Dataset Source
        # We pass RAW data here. No HTML strings. This prevents serialization crashes.
        dataset_source = []
        for i, row in df.iterrows():
            # If "Alle Spieler" is selected, don't plot the HCP line. We pass None.
            hcp_val = row['HCP'] if self.selected_player != "Alle Spieler" else None

            def safe_str(val): return str(val) if pd.notna(val) else "-"
            def safe_int(val): return int(val) if pd.notna(val) else "-"

            dataset_source.append([
                row['Datum'].strftime('%d.%m.%y'), # 0: Date
                safe_int(row['Brutto']),           # 1: Brutto
                hcp_val,                           # 2: HCP
                safe_str(row['Club']),             # 3: Club
                safe_str(row['Turnier']),          # 4: Turnier
                safe_str(row['Spieler_Name']),     # 5: Player
                safe_str(row.get('Spielmodus')),   # 6: Mode
                safe_int(row.get('Par')),          # 7: Par
                safe_int(row.get('uPar')),         # 8: uPar (raw)
                safe_int(row.get('cr')),           # 9: CR
            ])

        series = []

        # Series A: Brutto Bars (Mapped to Dataset)
        series.append({
            "name": "Brutto",
            "type": "bar",
            "encode": {"x": 0, "y": 1}, # X is Date(0), Y is Brutto(1)
            "yAxisIndex": 1, # Maps to right axis
            "itemStyle": {
                "color": "#88f088", # Safe solid color instead of JS gradient
                "borderRadius": [4, 4, 0, 0] # Rounded top corners
            },
            "label": {
                "show": True,
                "position": "inside",
                "color": "black",
                # Optimization: 10px is readable, bold helps, rotation prevents overlap on dense charts
                "fontSize": 10,
                "fontWeight": "bold",
                "rotate": 90,
                "formatter": "{@1}" # Renders the Brutto value from dimension 1
            },
        })

        # Series B: HCP Line (Mapped to Dataset, Conditional)
        if self.selected_player != "Alle Spieler":
            series.append({
                "name": "HCP",
                "type": "line",
                "encode": {"x": 0, "y": 2}, # X is Date(0), Y is HCP(2)
                "yAxisIndex": 0, # Maps to left axis
                "smooth": True, # Smoother lines are visually appealing
                "symbol": "circle",
                "symbolSize": 6,
                "itemStyle": {"color": "#1f77b4", "borderColor": "#fff", "borderWidth": 2},
                "lineStyle": {"width": 2, "color": "#1f77b4"},
            })

        # Series C: uPar Labels (Scatter Overlay)
        if upar_data:
             series.append({
                "name": "Über Par",
                "type": "scatter",
                "yAxisIndex": 1,
                "symbolSize": 1, # Invisible point
                "itemStyle": {"opacity": 0},
                "data": upar_data, # Directly provided [x_index, y_value, label_text]
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{@2}", # Reads the 3rd element (index 2) which is our "+val" string
                    "color": "#e53935", # Distinct red color for uPar
                    "fontSize": 10,
                    "fontWeight": "bold",
                    "distance": 3 # Closer to the bar
                },
                "z": 10, # Ensure it renders above bars
                "tooltip": {"show": False} # Disable tooltip for these text anchors
             })

        option = {
            "backgroundColor": "#ffffff",
            "dataset": {
                "source": dataset_source
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "backgroundColor": "rgba(255, 255, 255, 0.95)",
                "borderColor": "#e0e0e0",
                "borderWidth": 1,
                "padding": 12,
                "textStyle": {"color": "#333", "fontSize": 12},
                "extraCssText": "box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-radius: 8px;",

                # CLIENT-SIDE HTML GENERATION
                # We build the HTML entirely in JS using the raw dataset array.
                # This guarantees no "Python code" leaks and prevents JSON parsing crashes.
                 "formatter": """function (params) {
                    // Extract data from the first series that triggered the tooltip
                    var data = params[0].data || params[0].value;
                    if (!Array.isArray(data)) return '';

                    // Map to our dataset structure: [Date, Brutto, HCP, Club, Turnier, Player, Mode, Par, uPar, CR]
                    var date = data[0];
                    var brutto = data[1];
                    var club = data[3];
                    var turnier = data[4];
                    var player = data[5];
                    var mode = data[6];
                    var par = data[7];
                    var uparRaw = data[8];
                    var cr = data[9];

                    // Format uPar string on the client
                    var uparFmt = uparRaw;
                    if (uparRaw !== "-" && typeof uparRaw === "number") {
                        uparFmt = (uparRaw > 0 ? "+" : "") + uparRaw;
                    }

                    // Build elegant HTML table for tooltip
                    var html = '<div style="min-width: 180px;">';
                    html += '<div style="font-weight:bold; font-size:14px; border-bottom:1px solid #eee; padding-bottom:5px; margin-bottom:5px;">' + date + '</div>';
                    html += '<div style="font-size:11px; color:#666; margin-bottom:2px;">' + turnier + '</div>';
                    html += '<div style="font-size:11px; color:#666; margin-bottom:8px;">' + club + '</div>';

                    html += '<table style="width:100%; font-size:12px; line-height: 1.4;">';
                    html += '<tr><td style="color:#888;">Spieler</td><td style="text-align:right; font-weight:500;">' + player + '</td></tr>';
                    html += '<tr><td style="color:#888;">Modus</td><td style="text-align:right;">' + mode + '</td></tr>';
                    html += '<tr><td style="color:#888;">Brutto</td><td style="text-align:right; font-weight:bold; color:#28b028;">' + brutto + '</td></tr>';
                    html += '<tr><td style="color:#888;">Par</td><td style="text-align:right;">' + par + ' <span style="color:#e53935; font-size:10px;">(' + uparFmt + ')</span></td></tr>';
                    html += '<tr><td style="color:#888;">CR</td><td style="text-align:right;">' + cr + '</td></tr>';
                    html += '</table></div>';

                    return html;
                }"""
            },
            "legend": {
                "data": ["HCP", "Brutto"],
                "top": 0,
                "icon": "circle"
            },
            "grid": {
                "left": "3%",
                "right": "3%",
                "bottom": "18%", # Extra space for dataZoom and labels
                "top": "12%",
                "containLabel": True
            },
            "xAxis": {
                "type": "category",
                # Use dates list directly for xAxis data so ECharts knows the exact categories
                # This ensures the scatter plot aligns perfectly with the bars
                "data": dates,
                "axisTick": {"alignWithLabel": True},
                "axisLine": {"lineStyle": {"color": "#e0e0e0"}},
                "axisLabel": {
                    "color": "#666",
                    "fontSize": 10
                }
            },
            "yAxis": [
                {
                    "type": "value",
                    "name": "HCP",
                    "position": "left",
                    "nameTextStyle": {"color": "#1f77b4", "fontWeight": "bold"},
                    "axisLine": {"show": True, "lineStyle": {"color": "#1f77b4"}},
                    "axisLabel": {"color": "#1f77b4"},
                    "splitLine": {"show": True, "lineStyle": {"type": "dashed", "color": "#f0f0f0"}}
                },
                {
                    "type": "value",
                    "name": "Brutto",
                    "position": "right",
                    "nameTextStyle": {"color": "#28b028", "fontWeight": "bold"},
                    "axisLine": {"show": True, "lineStyle": {"color": "#28b028"}},
                    "axisLabel": {"color": "#28b028"},
                    "splitLine": {"show": False}
                }
            ],
            "dataZoom": [
                {
                    "type": "slider",
                    "show": True,
                    "xAxisIndex": [0],
                    "start": 0,
                    "end": 100,
                    "bottom": 10,
                    "borderColor": "transparent",
                    "backgroundColor": "#f9f9f9",
                    "fillerColor": "rgba(40, 176, 40, 0.2)",
                    "handleStyle": {"color": "#28b028"}
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

        # Handle initial zoom logic to show roughly the last 20 rounds
        total_bars = len(df)
        if total_bars > 20:
             start_pct = max(0, ((total_bars - 20) / total_bars) * 100)
             option["dataZoom"][0]["start"] = start_pct
             option["dataZoom"][1]["start"] = start_pct

        # Add MarkLines for Winter Breaks
        if mark_lines:
             if series:
                 series[0]["markLine"] = {
                     "symbol": "none",
                     "label": {"show": False},
                     "lineStyle": {"type": "solid", "color": "#ff9800", "width": 1.5, "opacity": 0.6},
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
                        GolfState.selected_player == "Alle Spieler",
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
                        rx.text(r["upar"], font_size="0.75em", font_weight="bold", color="red"),
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
                rx.table.body(rx.foreach(GolfState.player_summary_list, player_summary_row)),
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
            rx.text(GolfState.stats_title, font_size="0.8em", color="gray"),
            rx.hstack(
                stat_box("Birdies", GolfState.scoring_stats["birdie"], "#FFD700", "rgba(255, 215, 0, 0.1)"),
                stat_box("Eagles", GolfState.scoring_stats["eagle"], "#FF8C00", "rgba(255, 140, 0, 0.1)"),
                stat_box("Albatros", GolfState.scoring_stats["albatross"], "#8B00FF", "rgba(139, 0, 255, 0.1)"),
                stat_box("Ace", GolfState.scoring_stats["ace"], "#FF0000", "rgba(255, 0, 0, 0.1)"),
                width="100%", spacing="2", justify="between"
            ),
            rx.hstack(
                rx.text("Gesamt: ", font_size="0.9em", color="gray"),
                rx.text(GolfState.scoring_total, font_size="0.9em", font_weight="bold"),
                rx.spacer(),
                rx.text("Ø pro Runde: ", font_size="0.9em", color="gray"),
                rx.text(GolfState.scoring_average, font_size="0.9em", font_weight="bold"),
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
                    rx.text(GolfState.tournament_counts["total"], font_size="1.5em", font_weight="bold"),
                    align="center", spacing="0", padding="0.5em", border_radius="md", background_color="#f0f0f0",
                    width="100%"
                ),
                rx.vstack(
                    rx.text("Mit Ergebnis", font_size="0.8em", color="gray"),
                    rx.text(GolfState.tournament_counts["with_result"], font_size="1.5em", font_weight="bold",
                            color="green"),
                    align="center", spacing="0", padding="0.5em", border_radius="md",
                    background_color="rgba(0, 255, 0, 0.05)", width="100%"
                ),
                rx.vstack(
                    rx.text("Einzel", font_size="0.8em", color="gray"),
                    rx.text(GolfState.tournament_counts["single"], font_size="1.5em", font_weight="bold", color="blue"),
                    align="center", spacing="0", padding="0.5em", border_radius="md",
                    background_color="rgba(0, 0, 255, 0.05)", width="100%"
                ),
                rx.vstack(
                    rx.text("Vorgabewirksam", font_size="0.8em", color="gray"),
                    rx.text(GolfState.tournament_counts["vgw"], font_size="1.5em", font_weight="bold", color="purple"),
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
                    value=GolfState.password_input,
                    on_change=GolfState.set_password_input,
                    on_key_down=GolfState.handle_key_down,
                    placeholder="Passwort",
                    type="password",
                    width="250px",
                    aria_label="Passwort Eingabefeld",
                ),
                rx.cond(
                    GolfState.auth_error,
                    rx.text(GolfState.auth_error, color="red", font_size="0.9em"),
                ),
                rx.button(
                    "Anmelden",
                    on_click=GolfState.check_password,
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
                rx.button("Abmelden", on_click=GolfState.logout, size="2", variant="soft"),
                width="100%", padding_y="0.4em",
            ),

            # --- DATASET SELECTOR ---
            rx.hstack(
                rx.text("Datensatz:", font_size="0.9em", font_weight="bold"),
                rx.select(
                    GolfState.available_datasets,
                    value=GolfState.selected_dataset,
                    on_change=GolfState.set_selected_dataset,
                    width="100%", max_width="400px",
                    custom_attrs={"aria-label": "Datensatz Auswahl"}
                ),
                width="100%", padding_bottom="1em", align="center", spacing="2"
            ),

            rx.vstack(
                rx.hstack(
                    rx.text("Spieler:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(GolfState.player_options, value=GolfState.selected_player,
                              on_change=GolfState.set_selected_player, width="100%", max_width="400px",
                              custom_attrs={"aria-label": "Spieler Auswahl"}),
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Jahr:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(GolfState.year_options, value=GolfState.selected_year,
                              on_change=GolfState.set_selected_year, width="100%", max_width="400px",
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
                        value=GolfState.filter_turnierart,
                        on_change=GolfState.set_filter_turnierart,
                        width="180px",
                        custom_attrs={"aria-label": "Turnierart Auswahl"}
                    ),
                    align="center", spacing="2"
                ),
                rx.hstack(rx.checkbox(checked=GolfState.filter_brutto, on_change=GolfState.toggle_brutto),
                          rx.text("nur Turniere mit Ergebnis", font_size="0.9em"), align="center", spacing="1"),
                justify="center", width="100%", spacing="4", padding_bottom="1em", wrap="wrap"
            ),

            rx.box(
                rx.cond(
                    GolfState.filtered_df.empty,
                    rx.center(
                        rx.vstack(
                            rx.icon(tag="info", size=40, color="gray"),
                            rx.text("Keine Daten für diese Auswahl gefunden.", color="gray", font_weight="bold"),
                            rx.text("Bitte ändern Sie die Filterkriterien.", color="gray", font_size="0.9em"),
                            align="center", spacing="2"
                        ),
                        padding="40px", width="100%", height="100%"
                    ),
                    rx_echarts.echarts(
                        option=GolfState.echarts_option,
                        style={"width": "100%", "height": "100%"},
                    )
                ),
                width="100%", min_height="450px", height="60vh", border_radius="lg", border="1px solid #e0e0e0",
                background_color="white", box_shadow="sm", overflow="hidden"
            ),

            scoring_stats_card(),
            tournament_summary_card(),

            rx.vstack(
                rx.heading("Letzte 10 Runden", size="3"),
                rx.foreach(GolfState.last_five_rounds, round_card),
                width="100%", background_color="#fcfcfc", padding="1.2em", border_radius="lg",
                border="1px solid #e0e0e0", box_shadow="sm"
            ),

            # --- CONDITIONAL TABLE ---
            rx.cond(
                GolfState.selected_player == "Alle Spieler",
                player_summary_table(),
            ),

            width="100%", spacing="3", padding_bottom="3em"
        ),
        padding="1em", max_width="900px", margin="0 auto",
    )


def index():
    return rx.cond(GolfState.is_authenticated, dashboard(), login_form())


app = rx.App()
app.add_page(index, on_load=GolfState.on_load)
