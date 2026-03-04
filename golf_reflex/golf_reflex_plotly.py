import reflex as rx
import reflex_echarts as rx_echarts
import pandas as pd
import numpy as np
import os
import ast

MODULE_DIR: str = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR: str = os.path.join(os.path.dirname(MODULE_DIR), "assets")


# --- 1. DATA LOADING & PRE-PROCESSING ---
def load_and_prep_data(dataset_name: str = "AK50"):
    file_path = os.path.join(ASSETS_DIR, f"{dataset_name}.parquet")
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
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

    df[['Turnier', 'Club', 'Spielmodus']] = df[['Turnier', 'Club', 'Spielmodus']].fillna("-")

    df['Vorname'] = df['Vorname'].fillna('').astype(str).str.strip()
    df['Name'] = df['Name'].fillna('').astype(str).str.strip()

    mask_both = (df['Vorname'] != '') & (df['Name'] != '')
    mask_name_only = (df['Vorname'] == '') & (df['Name'] != '')
    mask_vorname_only = (df['Vorname'] != '') & (df['Name'] == '')

    df['Spieler_Name'] = 'Unknown'
    df.loc[mask_both, 'Spieler_Name'] = df.loc[mask_both, 'Name'] + ', ' + df.loc[mask_both, 'Vorname']
    df.loc[mask_name_only, 'Spieler_Name'] = df.loc[mask_name_only, 'Name']
    df.loc[mask_vorname_only, 'Spieler_Name'] = df.loc[mask_vorname_only, 'Vorname']

    if 'HCPRelevant' in df.columns:
        df['HCPRelevant'] = df['HCPRelevant'].astype(str).str.lower().str.strip()
        true_values = ['true', 'yes', '1', '1.0', 'ja', 't', 'y', 'j']
        df['HCPRelevant'] = df['HCPRelevant'].isin(true_values)
    else:
        print("WARNING: 'HCPRelevant' column not found, defaulting to False")
        df['HCPRelevant'] = False

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

                if not p_list or not s_list:
                    return [0, 0, 0, 0]

                birdies, eagles, albatross, aces = 0, 0, 0, 0
                for i in list(range(0, 9)) + list(range(10, 19)):
                    if i >= len(p_list) or i >= len(s_list):
                        continue
                    par_val, score_val = p_list[i], s_list[i]
                    if pd.isna(par_val) or pd.isna(score_val) or par_val == 0 or score_val == 0:
                        continue
                    diff = score_val - par_val
                    if score_val == 1:
                        aces += 1
                    if diff == -1:
                        birdies += 1
                    elif diff == -2 and score_val != 1:
                        eagles += 1
                    elif diff == -3 and score_val != 1:
                        albatross += 1
                return [birdies, eagles, albatross, aces]
            except:
                return [0, 0, 0, 0]

        stats_data = df.apply(calc_row_deltas, axis=1, result_type='expand')
        df[['cnt_birdie', 'cnt_eagle', 'cnt_albatross', 'cnt_ace']] = stats_data

    df['Jahr'] = df['Datum'].dt.year
    return df


def get_sort_key(name):
    if ',' in name:
        return name.split(',')[0].strip().lower()
    return name.lower()


APP_PASSWORD = os.environ.get("APP_PASSWORD", "nobi")


class GolfState(rx.State):
    is_authenticated: bool = False
    password_input: str = ""
    auth_error: str = ""

    filter_brutto: bool = False
    filter_turnierart: str = "Alle"
    selected_player: str = "Alle Spieler"
    selected_year: str = "2025"
    selected_dataset: str = "AK50"

    # DataFrames stay as private backend vars — NEVER exposed as @rx.var
    _data: pd.DataFrame = pd.DataFrame()

    # Chart window controls
    view_window_size: int = 50
    view_start_index: int = -1

    def on_load(self):
        self._reload_data()

    def _reload_data(self):
        self._data = load_and_prep_data(self.selected_dataset)
        self.view_start_index = -1
        if "Spieler_Name" in self._data.columns:
            if self.selected_player not in self.player_options:
                self.selected_player = "Alle Spieler"
        if "Jahr" in self._data.columns:
            years = self.year_options
            if self.selected_year not in years:
                self.selected_year = years[1] if len(years) > 1 else "Alle Jahre"

    # Private helper methods — return DataFrames for internal use only
    def _get_period_df(self) -> pd.DataFrame:
        if not self.is_authenticated or self._data.empty:
            return pd.DataFrame()
        df = self._data.copy()
        if self.selected_player != "Alle Spieler":
            df = df[df['Spieler_Name'] == self.selected_player]
        if self.selected_year != "Alle Jahre":
            df = df[df['Jahr'] == int(self.selected_year)]
        return df

    def _get_filtered_df(self) -> pd.DataFrame:
        df = self._get_period_df()
        if df.empty:
            return df
        if self.filter_brutto:
            df = df[(df['Brutto'].notna()) & (df['Brutto'] != 0)]
        if self.filter_turnierart == "Einzel":
            df = df[df['Spielmodus'].str.startswith('Einzel', na=False)]
        elif self.filter_turnierart == "Vorgabewirksam":
            df = df[df['HCPRelevant'] == True]
        return df.sort_values('Datum').reset_index(drop=True)

    # --- SERIALIZABLE @rx.var ONLY ---

    @rx.var
    def has_data(self) -> bool:
        return not self._get_filtered_df().empty

    @rx.var
    def total_filtered_rounds(self) -> int:
        return len(self._get_filtered_df())

    @rx.var
    def max_start_index(self) -> int:
        return max(0, self.total_filtered_rounds - self.view_window_size)

    @rx.var
    def actual_start_index(self) -> int:
        if self.view_start_index == -1:
            return self.max_start_index
        return max(0, min(self.view_start_index, self.max_start_index))

    def set_view_window_size(self, val: list[int]):
        self.view_window_size = int(val[0])
        if self.view_start_index != -1:
            self.view_start_index = min(self.view_start_index, self.max_start_index)

    def set_view_start_index(self, val: list[int]):
        self.view_start_index = int(val[0])

    def page_older(self):
        self.view_start_index = max(0, self.actual_start_index - self.view_window_size)

    def page_newer(self):
        self.view_start_index = min(self.max_start_index, self.actual_start_index + self.view_window_size)

    def set_selected_dataset(self, value: str):
        self.selected_dataset = value
        self._reload_data()

    @rx.var
    def available_datasets(self) -> list[str]:
        if not os.path.exists(ASSETS_DIR):
            return []
        files = [f for f in os.listdir(ASSETS_DIR) if f.endswith('.parquet')]
        return sorted([os.path.splitext(f)[0] for f in files])

    @rx.var
    def player_options(self) -> list[str]:
        if self._data.empty:
            return ["Alle Spieler"]
        return ["Alle Spieler"] + sorted(self._data['Spieler_Name'].unique().tolist(), key=get_sort_key)

    @rx.var
    def year_options(self) -> list[str]:
        if self._data.empty:
            return ["Alle Jahre"]
        return ["Alle Jahre"] + [str(y) for y in sorted(self._data['Jahr'].unique().tolist(), reverse=True)]

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
        if key == "Enter":
            return self.check_password()

    def logout(self):
        self.is_authenticated = False
        self.auth_error = ""
        self.password_input = ""

    # --- FILTERS ---
    def toggle_brutto(self, value: bool):
        self.filter_brutto = value
        self.view_start_index = -1

    def set_filter_turnierart(self, value: str):
        self.filter_turnierart = value
        self.view_start_index = -1

    def set_selected_player(self, value: str):
        self.selected_player = value
        self.view_start_index = -1

    def set_selected_year(self, value: str):
        self.selected_year = value
        self.view_start_index = -1

    # --- COMPUTED VARS (serializable only) ---

    @rx.var
    def tournament_counts(self) -> dict:
        df = self._get_period_df()
        if df.empty:
            return {"total": 0, "with_result": 0, "single": 0, "vgw": 0}
        return {
            "total": len(df),
            "with_result": len(df[(df['Brutto'].notna()) & (df['Brutto'] != 0)]),
            "single": len(df[df['Spielmodus'].str.startswith('Einzel', na=False)]),
            "vgw": len(df[df['HCPRelevant'] == True]),
        }

    @rx.var
    def stats_title(self) -> str:
        p = "alle Spieler" if self.selected_player == "Alle Spieler" else self.selected_player
        y = "allen Jahren" if self.selected_year == "Alle Jahre" else self.selected_year
        return f"Für {p} im Jahr {y}"

    @rx.var
    def scoring_stats(self) -> dict:
        df = self._get_filtered_df()
        if df.empty or 'cnt_birdie' not in df.columns:
            return {"birdie": 0, "eagle": 0, "albatross": 0, "ace": 0}
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
        count = self.total_filtered_rounds
        return f"{self.scoring_total / count:.1f}" if count > 0 else "0.0"

    @rx.var
    def last_five_rounds(self) -> list[dict]:
        df = self._get_filtered_df()
        if df.empty:
            return []
        rounds = []
        for _, r in df.tail(10).iloc[::-1].iterrows():
            upar_val = f"{int(r['uPar']):+d}" if not pd.isna(r.get('uPar')) else ""
            brutto_display = f"Brutto: {int(r['Brutto'])}" if pd.notna(r.get('Brutto')) else "Brutto: -"
            rounds.append({
                "date": r["Datum"].strftime('%d.%m.%y'),
                "club": str(r["Club"]),
                "turnier": str(r["Turnier"]),
                "hcp": f"HCP: {r['HCP']}",
                "upar": upar_val,
                "brutto": brutto_display,
                "player": str(r["Spieler_Name"]),
            })
        return rounds

    @rx.var
    def player_summary_list(self) -> list[dict]:
        df = self._get_filtered_df()
        if self.selected_player != "Alle Spieler" or df.empty:
            return []
        summary = df.groupby('Spieler_Name').agg(
            Rounds=('Turnier', 'count'),
            Avg_Brutto=('Brutto', 'mean'),
            Last_HCP=('HCP', 'last'),
        ).reset_index()
        summary['Avg_Brutto'] = summary['Avg_Brutto'].round(1)
        return summary.to_dict('records')

    @rx.var
    def echarts_option(self) -> dict:
        df = self._get_filtered_df()
        if df.empty:
            return {}

        df = df.copy()

        # Apply view window
        start = self.actual_start_index
        end = min(start + self.view_window_size, len(df))
        df_window = df.iloc[start:end].reset_index(drop=True)

        dates = df_window['Datum'].dt.strftime('%d.%m.%y').tolist()
        brutto_data = [int(v) for v in df_window['Brutto'].fillna(0)]
        hcp_data = df_window['HCP'].tolist()

        # Dynamic label font size: fewer bars = bigger font, more bars = smaller
        bar_font_size = max(7, min(13, int(180 / max(self.view_window_size, 1))))

        def build_tooltip(r) -> str:
            # Build all parts as local vars first — avoids nested quote issues
            date_str = r['Datum'].strftime('%d.%m.%y')
            turnier_str = str(r.get('Turnier', '-'))
            club_str = str(r.get('Club', '-'))
            spieler_str = str(r.get('Spieler_Name', '-'))
            modus_str = str(r.get('Spielmodus', '-'))
            brutto_str = str(int(r['Brutto'])) if pd.notna(r.get('Brutto')) else '-'
            hcp_raw = r.get('HCP')
            hcp_str = str(round(float(hcp_raw), 1)) if pd.notna(hcp_raw) else '-'
            par_val = r.get('Par')
            upar_val = r.get('uPar')
            if pd.notna(upar_val) and pd.notna(par_val):
                par_str = f"{int(par_val)} | {int(upar_val):+d}"
            elif pd.notna(par_val):
                par_str = str(int(par_val))
            else:
                par_str = '-'
            cr_raw = r.get('CR', r.get('cr'))
            cr_str = str(int(cr_raw)) if pd.notna(cr_raw) else '-'
            return (
                f"<div style='text-align:left;font-size:12px;line-height:1.6;'>"
                f"<b>{date_str}</b><br/>"
                f"{turnier_str}<br/>"
                f"{club_str}<br/>"
                f"<b>Spieler:</b> {spieler_str}<br/>"
                f"<b>HCP:</b> {hcp_str}<br/>"
                f"<b>Spielmodus:</b> {modus_str}<br/>"
                f"<b>Brutto:</b> {brutto_str}<br/>"
                f"<b>Par:</b> {par_str}<br/>"
                f"<b>CR:</b> {cr_str}"
                f"</div>"
            )

        tooltip_html = [build_tooltip(r) for _, r in df_window.iterrows()]


        # uPar: invisible bar series with labels above — mirrors Plotly mode='text' exactly.
        # Using bars (not scatter) guarantees category-axis alignment with Brutto bars.
        upar_label_data = []
        for i, (_, row) in enumerate(df_window.iterrows()):
            upar_val = row.get('uPar')
            brutto_val = int(row['Brutto']) if pd.notna(row.get('Brutto')) else 0
            if pd.notna(upar_val):
                upar_label_data.append({
                    "value": brutto_val,  # same height so label sits above the bar
                    "label": {
                        "show": True,
                        "formatter": f"{int(upar_val):+d}",
                        "position": "top",
                        "color": "#333",
                        "fontSize": bar_font_size,
                        "fontWeight": "bold",
                        "distance": 4,
                    },
                })
            else:
                upar_label_data.append({"value": brutto_val, "label": {"show": False}})

        # Year-boundary marklines: placed at first bar of each new year
        mark_lines = []
        if len(df_window) > 1:
            years = df_window['Datum'].dt.year.tolist()
            for i in range(1, len(years)):
                if years[i] != years[i - 1]:
                    mark_lines.append({"xAxis": dates[i]})

        series = []

        # Bar: Brutto — tooltip HTML stored per data item
        bar_data = [
            {
                "value": brutto_data[i],
                "tooltip": {"formatter": tooltip_html[i]},
            }
            for i in range(len(dates))
        ]
        bar_series = {
            "name": "Brutto",
            "type": "bar",
            "data": bar_data,
            "yAxisIndex": 1,
            "itemStyle": {"color": "rgba(84, 245, 66, 0.6)"},
            "label": {
                "show": True,
                "position": "inside",
                "color": "black",
                "fontSize": bar_font_size,
                "rotate": 90,
            },
        }
        if mark_lines:
            bar_series["markLine"] = {
                "symbol": "none",
                "label": {"show": False},
                "lineStyle": {"type": "dashed", "color": "red", "opacity": 0.4},
                "data": mark_lines,
            }
        series.append(bar_series)

        # Line: HCP (single player only)
        if self.selected_player != "Alle Spieler":
            series.append({
                "name": "HCP",
                "type": "line",
                "data": list(zip(dates, hcp_data)),
                "yAxisIndex": 0,
                "smooth": False,
                "symbol": "circle",
                "symbolSize": 5,
                "itemStyle": {"color": "royalblue"},
                "lineStyle": {"width": 1.5, "color": "royalblue"},
            })

        # uPar: invisible bar series — labels only, no fill, no tooltip, not in legend
        series.append({
            "name": "Über Par",
            "type": "bar",
            "yAxisIndex": 1,
            "data": upar_label_data,
            "barGap": "-100%",      # overlap exactly with Brutto bars
            "itemStyle": {"opacity": 0},
            "emphasis": {"disabled": True},
            "tooltip": {"show": False},
            "legendHoverLink": False,
            "silent": True,         # no mouse events at all
            "label": {
                "show": True,       # series-level show enables per-item label overrides
                "position": "top",
                "color": "#333",
                "fontSize": bar_font_size,
                "fontWeight": "bold",
            },
            "z": 10,
        })

        return {
            "tooltip": {
                "trigger": "item",
                "backgroundColor": "rgba(255,255,255,0.95)",
                "padding": 10,
                "textStyle": {"color": "#333"},
                "extraCssText": "box-shadow:0 2px 8px rgba(0,0,0,0.15);",
            },
            "legend": {
                "data": ["HCP", "Brutto"] if self.selected_player != "Alle Spieler" else ["Brutto"],
                "top": 0,
            },
            "grid": {
                "left": "3%",
                "right": "3%",
                "bottom": "5%",
                "top": "15%",
                "containLabel": True,
            },
            "xAxis": {
                "type": "category",
                "data": dates,
                "axisTick": {"alignWithLabel": True},
                "axisLabel": {"color": "#333", "rotate": 45, "fontSize": 10},
            },
            "yAxis": [
                {
                    "type": "value",
                    "name": "HCP",
                    "position": "left",
                    "axisLine": {"show": True, "lineStyle": {"color": "royalblue"}},
                    "splitLine": {"show": True, "lineStyle": {"type": "dashed"}},
                },
                {
                    "type": "value",
                    "name": "Brutto",
                    "position": "right",
                    "axisLine": {"show": True, "lineStyle": {"color": "green"}},
                    "splitLine": {"show": False},
                },
            ],
            "series": series,
        }


# --- UI COMPONENTS ---

def chart_controls():
    return rx.cond(
        GolfState.total_filtered_rounds > 0,
        rx.vstack(
            rx.hstack(
                rx.text("Zoom (Runden):", font_size="0.8em", width="120px", font_weight="bold"),
                rx.slider(
                    value=[GolfState.view_window_size],
                    min=10,
                    max=100,
                    on_change=GolfState.set_view_window_size,
                    width="100%",
                ),
                rx.text(GolfState.view_window_size, font_size="0.8em", width="30px", text_align="right"),
                width="100%",
                align="center",
            ),
            rx.hstack(
                rx.button(
                    "<< Ältere",
                    on_click=GolfState.page_older,
                    size="1",
                    variant="soft",
                    disabled=GolfState.actual_start_index == 0,
                ),
                rx.slider(
                    value=[GolfState.actual_start_index],
                    min=0,
                    max=GolfState.max_start_index,
                    on_change=GolfState.set_view_start_index,
                    width="100%",
                ),
                rx.button(
                    "Neuere >>",
                    on_click=GolfState.page_newer,
                    size="1",
                    variant="soft",
                    disabled=GolfState.actual_start_index == GolfState.max_start_index,
                ),
                width="100%",
                align="center",
                spacing="3",
            ),
            width="100%",
            padding="1em",
            background_color="#f8f9fa",
            border_radius="md",
            border="1px solid #e0e0e0",
            margin_bottom="0.5em",
        ),
        rx.box(),
    )


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
                    align_items="start",
                    spacing="0",
                ),
                rx.spacer(),
                rx.text(
                    r["turnier"],
                    font_size="0.65em",
                    font_style="italic",
                    color="gray",
                    max_width="200px",
                    text_align="center",
                ),
                rx.spacer(),
                rx.vstack(
                    rx.hstack(
                        rx.text(r["hcp"], font_size="0.7em", color="royalblue"),
                        rx.text(r["upar"], font_size="0.75em", font_weight="bold", color="blue"),
                        spacing="2",
                    ),
                    rx.text(r["brutto"], font_weight="bold", color="green", font_size="0.75em"),
                    align_items="end",
                    spacing="0",
                ),
                width="100%",
                align_items="center",
            ),
            border_bottom="1px solid #EDEDED",
            padding_y="0.5em",
            width="100%",
        ),
        width="100%",
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
                width="100%",
                variant="surface",
            ),
        ),
        width="100%",
        padding="1em",
        background_color="#fcfcfc",
        border_radius="lg",
        border="1px solid #e0e0e0",
        margin_top="1em",
    )


def scoring_stats_card():
    def stat_box(label, value, color, bg_color):
        return rx.vstack(
            rx.text(label, font_size="0.8em", color="gray"),
            rx.text(value, font_size="1.8em", font_weight="bold", color=color),
            align="center",
            spacing="0",
            padding="0.8em",
            border_radius="md",
            background_color=bg_color,
            width="100%",
            border=f"1px solid {color}4D",
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
                width="100%",
                spacing="2",
                justify="between",
            ),
            rx.hstack(
                rx.text("Gesamt: ", font_size="0.9em", color="gray"),
                rx.text(GolfState.scoring_total, font_size="0.9em", font_weight="bold"),
                rx.spacer(),
                rx.text("Ø pro Runde: ", font_size="0.9em", color="gray"),
                rx.text(GolfState.scoring_average, font_size="0.9em", font_weight="bold"),
                width="100%",
                justify="between",
                padding_top="0.5em",
            ),
            width="100%",
            padding="1em",
            background_color="#fcfcfc",
            border_radius="lg",
            border="1px solid #e0e0e0",
            box_shadow="sm",
        ),
        width="100%",
    )


def tournament_summary_card():
    return rx.box(
        rx.vstack(
            rx.heading("Turnier Übersicht", size="3"),
            rx.grid(
                rx.vstack(
                    rx.text("Gesamt", font_size="0.8em", color="gray"),
                    rx.text(GolfState.tournament_counts["total"], font_size="1.5em", font_weight="bold"),
                    align="center",
                    spacing="0",
                    padding="0.5em",
                    border_radius="md",
                    background_color="#f0f0f0",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Mit Ergebnis", font_size="0.8em", color="gray"),
                    rx.text(
                        GolfState.tournament_counts["with_result"],
                        font_size="1.5em",
                        font_weight="bold",
                        color="green",
                    ),
                    align="center",
                    spacing="0",
                    padding="0.5em",
                    border_radius="md",
                    background_color="rgba(0, 255, 0, 0.05)",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Einzel", font_size="0.8em", color="gray"),
                    rx.text(
                        GolfState.tournament_counts["single"],
                        font_size="1.5em",
                        font_weight="bold",
                        color="blue",
                    ),
                    align="center",
                    spacing="0",
                    padding="0.5em",
                    border_radius="md",
                    background_color="rgba(0, 0, 255, 0.05)",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Vorgabewirksam", font_size="0.8em", color="gray"),
                    rx.text(
                        GolfState.tournament_counts["vgw"],
                        font_size="1.5em",
                        font_weight="bold",
                        color="purple",
                    ),
                    align="center",
                    spacing="0",
                    padding="0.5em",
                    border_radius="md",
                    background_color="rgba(128, 0, 128, 0.05)",
                    width="100%",
                ),
                columns="2",
                spacing="2",
                width="100%",
            ),
            width="100%",
            padding="1em",
            background_color="#fcfcfc",
            border_radius="lg",
            border="1px solid #e0e0e0",
        ),
        width="100%",
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
                spacing="1",
                align="center",
            ),
            padding="3em",
            border="1px solid #e0e0e0",
            border_radius="lg",
            background_color="white",
            box_shadow="lg",
        ),
        width="100vw",
        height="100vh",
        background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    )


def dashboard():
    return rx.box(
        rx.vstack(
            # Header
            rx.hstack(
                rx.heading("🏌️ Golf Performance", size="6"),
                rx.spacer(),
                rx.button("Abmelden", on_click=GolfState.logout, size="2", variant="soft"),
                width="100%",
                padding_y="0.4em",
            ),

            # Dataset selector
            rx.hstack(
                rx.text("Datensatz:", font_size="0.9em", font_weight="bold"),
                rx.select(
                    GolfState.available_datasets,
                    value=GolfState.selected_dataset,
                    on_change=GolfState.set_selected_dataset,
                    width="100%",
                    max_width="400px",
                    custom_attrs={"aria-label": "Datensatz Auswahl"},
                ),
                width="100%",
                padding_bottom="1em",
                align="center",
                spacing="2",
            ),

            # Player & year filters
            rx.vstack(
                rx.hstack(
                    rx.text("Spieler:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(
                        GolfState.player_options,
                        value=GolfState.selected_player,
                        on_change=GolfState.set_selected_player,
                        width="100%",
                        max_width="400px",
                        custom_attrs={"aria-label": "Spieler Auswahl"},
                    ),
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Jahr:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(
                        GolfState.year_options,
                        value=GolfState.selected_year,
                        on_change=GolfState.set_selected_year,
                        width="100%",
                        max_width="400px",
                        custom_attrs={"aria-label": "Jahr Auswahl"},
                    ),
                    width="100%",
                ),
                width="100%",
                spacing="2",
                padding_bottom="1em",
            ),

            # Turnierart & Brutto filter
            rx.hstack(
                rx.hstack(
                    rx.text("Turnierart:", font_size="0.9em", font_weight="bold"),
                    rx.select(
                        ["Alle", "Einzel", "Vorgabewirksam"],
                        value=GolfState.filter_turnierart,
                        on_change=GolfState.set_filter_turnierart,
                        width="180px",
                        custom_attrs={"aria-label": "Turnierart Auswahl"},
                    ),
                    align="center",
                    spacing="2",
                ),
                rx.hstack(
                    rx.checkbox(checked=GolfState.filter_brutto, on_change=GolfState.toggle_brutto),
                    rx.text("nur Turniere mit Ergebnis", font_size="0.9em"),
                    align="center",
                    spacing="1",
                ),
                justify="center",
                width="100%",
                spacing="4",
                padding_bottom="1em",
                wrap="wrap",
            ),

            # Chart window controls
            chart_controls(),

            # Chart
            rx.box(
                rx.cond(
                    GolfState.has_data,
                    rx_echarts.echarts(
                        option=GolfState.echarts_option,
                        style={"width": "100%", "height": "100%"},
                    ),
                    rx.text("Keine Daten", color="gray", padding="40px", text_align="center"),
                ),
                width="100%",
                min_height="450px",
                height="55vh",
                border_radius="md",
                border="1px solid #e0e0e0",
                background_color="white",
            ),

            scoring_stats_card(),
            tournament_summary_card(),

            # Recent rounds
            rx.vstack(
                rx.heading("Letzte Runden", size="3"),
                rx.foreach(GolfState.last_five_rounds, round_card),
                width="100%",
                background_color="#fcfcfc",
                padding="1.2em",
                border_radius="lg",
                border="1px solid #e0e0e0",
                box_shadow="sm",
            ),

            # Player summary table (only when viewing all players)
            rx.cond(
                GolfState.selected_player == "Alle Spieler",
                player_summary_table(),
            ),

            width="100%",
            spacing="3",
        ),
        padding="1em",
        max_width="800px",
        margin="0 auto",
    )


def index():
    return rx.cond(GolfState.is_authenticated, dashboard(), login_form())


app = rx.App()
app.add_page(index, on_load=GolfState.on_load)
