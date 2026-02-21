import reflex as rx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import ast
from typing import List, Optional

MODULE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_PATH: str = os.path.join(os.path.dirname(MODULE_DIR), "assets", "data.csv")


# --- 1. DATA LOADING & ROBUST PRE-PROCESSING ---
# --- REPLACE THIS FUNCTION TO DEBUG ---
def load_and_prep_data():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
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
        print("FOUND 'HCPRelevant' COLUMN.")
        print("UNIQUE VALUES BEFORE CLEANING:", df['HCPRelevant'].unique())  # <--- LOOK AT THIS IN TERMINAL

        # Convert to string and clean
        df['HCPRelevant'] = df['HCPRelevant'].astype(str).str.lower().str.strip()

        # Expanded Map for German/English/Numbers
        true_values = ['true', 'yes', '1', '1.0', 'ja', 't', 'y', 'j']
        df['HCPRelevant'] = df['HCPRelevant'].isin(true_values)

        print(f"Total VGW found after cleaning: {len(df[df['HCPRelevant'] == True])}") # <--- CHECK THIS COUNT
    else:
        print("ERROR: 'HCPRelevant' column NOT found in CSV!")
        # Check if it has a different case
        possible_cols = [c for c in df.columns if 'hcp' in c.lower()]
        print(f"Did you mean one of these? {possible_cols}")
        df['HCPRelevant'] = False

    # --- STATS CALCULATION ---
    if 'par' in df.columns and 'score' in df.columns:
        def calc_row_deltas(row):
            try:
                p_list = ast.literal_eval(row['par']) if isinstance(row['par'], str) else []
                s_list = ast.literal_eval(row['score']) if isinstance(row['score'], str) else []

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
# Initialize Data Globally
df0 = load_and_prep_data()


# Helper for sorting
def get_sort_key(name):
    if ',' in name: return name.split(',')[0].strip().lower()
    return name.lower()


unique_players = sorted(df0['Spieler_Name'].unique().tolist(), key=get_sort_key) if not df0.empty else []
unique_years = sorted(df0['Jahr'].unique().tolist(), reverse=True) if not df0.empty else []
APP_PASSWORD = "nobi"


class GolfState(rx.State):
    is_authenticated: bool = False
    password_input: str = ""
    auth_error: str = ""

    # Filter states
    filter_brutto: bool = False
    filter_turnierart: str = "Alle"
    selected_player: str = "Alle Spieler"
    selected_year: str = "2025"

    @rx.var
    def player_options(self) -> list[str]:
        return ["Alle Spieler"] + unique_players

    @rx.var
    def year_options(self) -> list[str]:
        return ["Alle Jahre"] + [str(year) for year in unique_years]

    # --- AUTH ---
    def set_password_input(self, value: str):
        self.password_input = value

    def check_password(self):
        if self.password_input == APP_PASSWORD:
            self.is_authenticated = True
            self.auth_error = ""
            self.password_input = ""
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
        if not self.is_authenticated: return pd.DataFrame()

        df = df0.copy()

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
            df = df[df['Spielmodus'].str.startswith('Einzel', na=False)]
        elif self.filter_turnierart == "Vorgabewirksam":
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
    def figure(self) -> go.Figure:
        if self.filtered_df.empty:
            return go.Figure()

        df = self.filtered_df.copy()
        df['ContinuousIndex'] = range(len(df))

        # Safely grab Par (handling lowercase/uppercase) and CR columns
        par_series = df['Par'] if 'Par' in df.columns else (
            df['par'] if 'par' in df.columns else pd.Series(["-"] * len(df)))
        cr_series = df['CR'] if 'CR' in df.columns else pd.Series(["-"] * len(df))

        # Safely grab the aggregate Par and CR columns
        par_series = df['Par'] if 'Par' in df.columns else pd.Series(["-"] * len(df))
        cr_series = df['CR'] if 'CR' in df.columns else pd.Series(["-"] * len(df))
        print(df.head(2))
        df['HoverText'] = df.apply(lambda r: (
            f"<b>{r['Datum'].strftime('%d.%m.%y')}</b><br>"
            f"{r['Turnier']}<br>"
            f"{r['Club']}<br>"
            f"<b>Spieler:</b> {r['Spieler_Name']}<br>"
            f"<b>Spielmodus:</b> {r.get('Spielmodus', 'N/A')}<br>"
            f"<b>Brutto:</b> {int(r['Brutto'])}<br>"
            f"<b>Par:</b> {f'{int(r.get('Par'))} | {int(r.get('uPar', 0)):+,d}' if not pd.isna(r.get('uPar', None)) else r.get('Par')}<br>"
            f"<b>CR:</b> {int(r['cr'])}<br>"





            #    f"<b>Date:</b> {row['Datum'].strftime('%d-%m-%y')}<br>"
            #    f"<b>Turnier:</b> {row.get('Turnier', 'N/A')}<br>"
            #    f"<b>Club:</b> {row.get('Club', 'N/A')}<br>"
            #    f"<b>Spielmodus:</b> {row.get('Spielmodus', 'N/A')}<br>"
            #    f"<b>HCP:</b> {row.get('HCP', 0)}<br>"
            #    f"<b>Brutto:</b> {int(row.get('Brutto', 0))}<br>"
            #    f"<b>Par:</b> {f'{row.get('Par')} / {int(row.get('uPar', 0)):+,d}' if not pd.isna(row.get('uPar', None)) else row.get('Par')}"

        ), axis=1)

        fig = go.Figure()

        # 2. HCP Line (Hidden if Alle Spieler)
        if self.selected_player != "Alle Spieler":
            fig.add_trace(go.Scatter(
                x=df['ContinuousIndex'], y=df['HCP'],
                mode='lines+markers',
                name='HCP',
                line=dict(width=1.5, color='royalblue'),
                marker=dict(size=4, color='royalblue'),
                text=df['HoverText'],
                hoverinfo='text',
            ))

        # 3. Brutto Bars
        fig.add_trace(go.Bar(
            x=df['ContinuousIndex'], y=df['Brutto'],
            name='Brutto',
            marker_color='rgba(84, 245, 66, 0.6)',
            text=df['Brutto'].astype(int),
            textposition='inside',
            textfont=dict(size=18, color='black'),
            hovertext=df['HoverText'],
            hoverinfo='text',
            yaxis='y2',
            marker_line_width=0,
        ))

        # 4. uPar Labels
        df_upar = df.dropna(subset=['uPar'])
        if not df_upar.empty:
            upar_text = df_upar['uPar'].astype(int).apply(lambda x: f"{x:+}")
            fig.add_trace(go.Scatter(
                x=df_upar['ContinuousIndex'], y=df_upar['Brutto'],
                mode='text',
                name='Über Par',
                text=upar_text,
                textfont=dict(size=9, color='black', weight='bold'),
                hoverinfo='skip',
                yaxis='y2',
                textposition='top center',
                cliponaxis=False
            ))

        # 5. Winter Breaks
        if len(df) > 1:
            date_diffs = df['Datum'].diff().dt.days
            gap_indices = df.index[date_diffs > 50].tolist()
            for idx in gap_indices:
                if idx > 0:
                    try:
                        pos_current = df.loc[idx, 'ContinuousIndex']
                        pos_prev = df.loc[df.index[df.index.get_loc(idx) - 1], 'ContinuousIndex']
                        pos = (pos_current + pos_prev) / 2
                        fig.add_vline(x=pos, line_dash="dash", line_color="red", opacity=0.4)
                    except:
                        pass

        total_bars = len(df)
        initial_min = max(-0.5, total_bars - 20.5)
        initial_max = total_bars - 0.5
        abs_min = -0.5
        abs_max = total_bars - 0.5

        fig.update_layout(
            template='plotly_white',
            hovermode='closest',
            dragmode='pan',
            xaxis=dict(
                tickvals=df['ContinuousIndex'],
                ticktext=df['Datum'].dt.strftime('%d.%m.%y'),
                range=[initial_min, initial_max],
                minallowed=abs_min,
                maxallowed=abs_max,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
            ),
            yaxis=dict(fixedrange=True, title="HCP", side='left'),
            yaxis2=dict(overlaying='y', side='right', fixedrange=True, title="Brutto", showgrid=False),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.1),
        )

        return fig


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
            rx.vstack(
                rx.hstack(
                    rx.text("Spieler:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(GolfState.player_options, value=GolfState.selected_player,
                              on_change=GolfState.set_selected_player, width="100%", max_width="400px"),
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Jahr:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(GolfState.year_options, value=GolfState.selected_year,
                              on_change=GolfState.set_selected_year, width="100%", max_width="400px"),
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
                        width="180px"
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
                    rx.text("Keine Daten", color="gray", padding="40px", text_align="center"),
                    rx.plotly(
                        data=GolfState.figure,
                        use_resize_handler=True,
                        style={"width": "100%", "height": "100%"},
                        config={"displayModeBar": False, "scrollZoom": True}
                    )
                ),
                width="100%", min_height="450px", height="55vh", border_radius="md", border="1px solid #e0e0e0",
                background_color="white"
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

            width="100%", spacing="3",
        ),
        padding="1em", max_width="800px", margin="0 auto",
    )


def index():
    return rx.cond(GolfState.is_authenticated, dashboard(), login_form())


app = rx.App()
app.add_page(index)