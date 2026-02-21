import reflex as rx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from typing import List, Optional

print('Hello')

# --- DATA LOADING & PREPARATION ---
MODULE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_PATH: str = os.path.join(os.path.dirname(MODULE_DIR), "assets", "data.csv")

# Load data once
try:
    df0 = pd.read_csv(DATA_PATH)
    df0['Datum'] = pd.to_datetime(df0['Datum'])
    df0 = df0.sort_values('Datum')
    # Fill string NaNs
    df0[['Turnier', 'Club', 'Spielmodus']] = df0[['Turnier', 'Club', 'Spielmodus']].fillna("-")

    # Create combined player name: "Lastname, Firstname"
    df0['Vorname'] = df0['Vorname'].fillna('').astype(str).str.strip()
    df0['Name'] = df0['Name'].fillna('').astype(str).str.strip()

    # Combine into "Lastname, Firstname" format
    df0['Spieler_Name'] = df0.apply(
        lambda x: f"{x['Name']}, {x['Vorname']}"
        if x['Vorname'] and x['Name']
        else x['Name'] if x['Name']
        else x['Vorname'],
        axis=1
    )

    # Clean up
    df0['Spieler_Name'] = df0['Spieler_Name'].str.strip()
    df0['Spieler_Name'] = df0['Spieler_Name'].replace({'': 'Unknown', 'nan': 'Unknown'})
except Exception as e:
    print(f"Error loading data: {e}")
    # Create dummy data if file is missing to prevent crash on startup
    df0 = pd.DataFrame(
        columns=['Datum', 'Turnier', 'Club', 'Spielmodus', 'Vorname', 'Name', 'Spieler_Name', 'Jahr', 'Brutto', 'HCP',
                 'uPar', 'par', 'score'])


# Get unique players for dropdown - sort alphabetically by last name
def get_sort_key(name):
    """Extract last name for sorting (part before comma if exists)"""
    if ',' in name:
        return name.split(',')[0].strip().lower()
    return name.lower()


if not df0.empty:
    unique_players = sorted(df0['Spieler_Name'].unique().tolist(), key=get_sort_key)
    # Get unique years for year filter
    df0['Jahr'] = df0['Datum'].dt.year
    unique_years = sorted(df0['Jahr'].unique().tolist(), reverse=True)
else:
    unique_players = []
    unique_years = []

print(f"Found {len(unique_players)} unique players")

# Password for simple authentication
APP_PASSWORD = "nobi"  # Change this to your desired password


# --- STATE MANAGEMENT ---
class GolfState(rx.State):
    # Authentication state
    is_authenticated: bool = False
    password_input: str = ""
    auth_error: str = ""

    # Filter states
    filter_brutto: bool = False
    filter_einzel: bool = False
    selected_player: str = "Sanchez, Antonio"
    selected_year: str = "Alle Jahre"

    # Force refresh counter
    refresh_counter: int = 0

    @rx.var
    def player_options(self) -> list[str]:
        """Get list of players for dropdown"""
        return ["Alle Spieler"] + unique_players

    @rx.var
    def year_options(self) -> list[str]:
        """Get list of years for dropdown"""
        return ["Alle Jahre"] + [str(year) for year in unique_years]

    @rx.var
    def stats_title(self) -> str:
        """Generate title for statistics card"""
        player_text = "alle Spieler" if self.selected_player == "Alle Spieler" else self.selected_player
        year_text = "allen Jahren" if self.selected_year == "Alle Jahre" else self.selected_year
        return f"Für {player_text} im Jahr {year_text}"

    @rx.var
    def scoring_total(self) -> int:
        """Calculate total scoring stats"""
        stats = self.scoring_stats
        return stats["birdie"] + stats["eagle"] + stats["albatross"] + stats["ace"]

    @rx.var
    def scoring_average(self) -> str:
        """Calculate average scoring stats"""
        stats = self.scoring_stats
        total = stats["birdie"] + stats["eagle"] + stats["albatross"] + stats["ace"]
        df_len = len(self.filtered_df)
        if df_len > 0:
            avg = total / df_len
            return f"{avg:.1f}"
        return "0.0"

    def check_password(self):
        """Check if the entered password is correct"""
        if self.password_input == APP_PASSWORD:
            self.is_authenticated = True
            self.auth_error = ""
            self.password_input = ""
            print("Authentication successful")
        else:
            self.auth_error = "Falsches Passwort"
            print("Authentication failed")

    def logout(self):
        """Log out the user"""
        self.is_authenticated = False
        self.password_input = ""
        self.auth_error = ""
        print("User logged out")

    def toggle_brutto(self, value: bool):
        if not self.is_authenticated: return
        self.filter_brutto = value
        self.refresh_counter += 1

    def toggle_einzel(self, value: bool):
        if not self.is_authenticated: return
        self.filter_einzel = value
        self.refresh_counter += 1

    def set_selected_player(self, value: str):
        if not self.is_authenticated: return
        self.selected_player = value
        self.refresh_counter += 1

    def set_selected_year(self, value: str):
        if not self.is_authenticated: return
        self.selected_year = value
        self.refresh_counter += 1

    @staticmethod
    def parse_list_string(list_str: str) -> Optional[List[float]]:
        """Parse string representation of list to actual list of floats"""
        if pd.isna(list_str) or not isinstance(list_str, str):
            return None
        try:
            list_str = list_str.strip("[]")
            if not list_str: return []
            items = list_str.split(',')
            result = []
            for item in items:
                item = item.strip()
                if item.lower() == 'nan':
                    result.append(float('nan'))
                else:
                    result.append(float(item))
            return result
        except Exception as e:
            return None

    @staticmethod
    def calculate_deltas(par_list: List[float], score_list: List[float]) -> dict:
        """Calculate birdies, eagles, albatross, and aces"""
        if not par_list or not score_list:
            return {"birdie": 0, "eagle": 0, "albatross": 0, "ace": 0}

        min_length = min(len(par_list), len(score_list), 19)
        birdie_count = 0
        eagle_count = 0
        albatross_count = 0
        ace_count = 0
        indices_to_check = list(range(0, 9)) + list(range(10, 19))

        for idx in indices_to_check:
            if idx >= min_length: continue
            par = par_list[idx]
            score = score_list[idx]
            if pd.isna(par) or pd.isna(score): continue

            delta = score - par
            if score == 1: ace_count += 1
            if delta == -3:
                albatross_count += 1
            elif delta == -2:
                eagle_count += 1
            elif delta == -1:
                birdie_count += 1

        return {
            "birdie": birdie_count,
            "eagle": eagle_count,
            "albatross": albatross_count,
            "ace": ace_count
        }

    @rx.var
    def filtered_df(self) -> pd.DataFrame:
        if not self.is_authenticated: return pd.DataFrame()
        # Triggers for reactivity
        _ = self.filter_brutto
        _ = self.filter_einzel
        _ = self.selected_player
        _ = self.selected_year
        _ = self.refresh_counter

        df = df0.copy()
        if self.selected_player != "Alle Spieler":
            df = df[df['Spieler_Name'] == self.selected_player]
        if self.selected_year != "Alle Jahre":
            year = int(self.selected_year)
            df = df[df['Jahr'] == year]
        if self.filter_brutto:
            df = df[df['Brutto'] != 0]
        if self.filter_einzel:
            mask = df['Spielmodus'].str.startswith('Einzel', na=False)
            df = df[mask]
        return df.sort_values('Datum').reset_index(drop=True)

    @rx.var
    def scoring_stats(self) -> dict:
        """Calculate birdie, eagle, albatross, and ace statistics"""
        if not self.is_authenticated: return {"birdie": 0, "eagle": 0, "albatross": 0, "ace": 0}
        df = self.filtered_df
        if 'par' not in df.columns or 'score' not in df.columns:
            return {"birdie": 0, "eagle": 0, "albatross": 0, "ace": 0}

        total_birdies = 0
        total_eagles = 0
        total_albatross = 0
        total_aces = 0

        for _, row in df.iterrows():
            par_list = self.parse_list_string(row['par'])
            score_list = self.parse_list_string(row['score'])
            if par_list is not None and score_list is not None:
                deltas = self.calculate_deltas(par_list, score_list)
                total_birdies += deltas["birdie"]
                total_eagles += deltas["eagle"]
                total_albatross += deltas["albatross"]
                total_aces += deltas["ace"]

        return {
            "birdie": total_birdies,
            "eagle": total_eagles,
            "albatross": total_albatross,
            "ace": total_aces
        }

    @rx.var
    def tournament_counts(self) -> dict:
        """Calculate tournament totals based on filters"""
        df = self.filtered_df
        if df.empty:
            return {"total": 0, "with_result": 0, "single": 0}

        return {
            "total": len(df),
            "with_result": len(df[df['Brutto'] != 0]),
            "single": len(df[df['Spielmodus'].str.startswith('Einzel', na=False)])
        }

    @rx.var
    def last_five_rounds(self) -> list[dict]:
        if not self.is_authenticated: return []
        df = self.filtered_df
        if df.empty: return []
        recent = df.tail(5).iloc[::-1].copy()
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
    def figure(self) -> go.Figure:
        if not self.is_authenticated: return go.Figure()
        # Triggers for reactivity
        _ = self.filtered_df
        _ = self.refresh_counter
        df = self.filtered_df

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(
                    text="Keine Daten für die gewählten Filter",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )],
                margin=dict(l=10, r=10, t=50, b=10)
            )
            return fig

        df['ContinuousIndex'] = range(len(df))
        df['HoverText'] = df.apply(lambda r: (
            f"<b>{r['Datum'].strftime('%d.%m.%y')}</b><br>"
            f"{r['Turnier']}<br>"
            f"Spieler: {r['Spieler_Name']}<br>"
            f"uPar: {f'{int(r.get('uPar', 0)):+d}' if not pd.isna(r.get('uPar')) else 'N/A'}<br>"
            f"Brutto: {int(r['Brutto'])}"
        ), axis=1)

        fig = go.Figure()

        # HCP Line
        fig.add_trace(go.Scatter(
            x=df['ContinuousIndex'], y=df['HCP'],
            mode='lines+markers',
            name='<span style="color:royalblue">● Handicap</span>',
            line=dict(width=1.5, color='royalblue'),
            marker=dict(size=4, color='royalblue'),
            text=df['HoverText'],
            hoverinfo='text',
            connectgaps=False,
        ))

        # Brutto Bars
        fig.add_trace(go.Bar(
            x=df['ContinuousIndex'], y=df['Brutto'],
            name='<span style="color:#54f542">●</span> Brutto Punkte',
            marker_color='rgba(84, 245, 66, 0.6)',
            hovertext=df['HoverText'],
            hoverinfo='text',
            yaxis='y2',
            marker_line_width=0,
        ))

        # Winter break lines (Improved: Subtle & Background)
        if len(df) > 1:
            date_diffs = df['Datum'].diff().dt.days
            gap_indices = df.index[date_diffs > 50].tolist()
            for idx in gap_indices:
                if idx > 0:
                    pos = (df.loc[idx - 1, 'ContinuousIndex'] + df.loc[idx, 'ContinuousIndex']) / 2
                    fig.add_vline(
                        x=pos,
                        line_dash="dot",
                        line_color="#cccccc",
                        opacity=0.5,
                        layer="below"
                    )

        # Set initial view
        total_bars = len(df)
        if total_bars > 20:
            initial_range = [max(-0.5, total_bars - 20.5), total_bars - 0.5]
        else:
            initial_range = [-0.5, total_bars - 0.5]

        plot_title = f"Golf Performance"
        if self.selected_player != "Alle Spieler":
            plot_title = f"Golf Performance - {self.selected_player}"

        fig.update_layout(
            template='plotly_white',
            hovermode='closest',
            dragmode='pan',  # Standard for mobile scrolling

            xaxis=dict(
                tickvals=df['ContinuousIndex'],
                ticktext=df['Datum'].dt.strftime('%d.%m.%y'),
                type='linear',  # Crucial for zoom!
                fixedrange=False,  # Allows horizontal zoom
                tickangle=45,
                tickfont=dict(size=9),
                range=initial_range,
                automargin=True,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=0.5,
            ),
            yaxis=dict(
                fixedrange=True,  # Lock vertical zoom
                side='left',
                automargin=True,
                title="HCP",
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=0.5,
            ),
            yaxis2=dict(
                overlaying='y',
                side='right',
                showgrid=False,
                fixedrange=True,  # Lock vertical zoom
                automargin=True,
                title="Brutto Punkte",
            ),
            margin=dict(l=10, r=10, t=40, b=10),  # Optimized for mobile
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0,
                y=1.1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
                font=dict(size=10)
            ),
            autosize=True,
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
                        rx.text(
                            r["player"],
                            font_size="0.7em",
                            color="#2c5282",
                            font_weight="bold"
                        ),
                    ),
                    align_items="start",
                    spacing="0"
                ),
                rx.spacer(),
                rx.text(
                    r["turnier"],
                    font_size="0.65em",
                    font_style="italic",
                    color="gray",
                    max_width="200px",
                    text_align="center"
                ),
                rx.spacer(),
                rx.vstack(
                    rx.hstack(
                        rx.text(r["hcp"], font_size="0.7em", color="royalblue"),
                        rx.text(r["upar"], font_size="0.75em", font_weight="bold", color="blue"),
                        spacing="2"
                    ),
                    rx.text(r["brutto"], font_weight="bold", color="green", font_size="0.75em"),
                    align_items="end",
                    spacing="0"
                ),
                width="100%",
                align_items="center"
            ),
            border_bottom="1px solid #EDEDED",
            padding_y="0.5em",
            width="100%"
        ),
        width="100%"
    )


def scoring_stats_card():
    return rx.box(
        rx.vstack(
            rx.heading("Scoring Statistiken", size="3"),
            rx.text(
                GolfState.stats_title,
                font_size="0.8em",
                color="gray"
            ),
            rx.hstack(
                # Albatross
                rx.vstack(
                    rx.text("Albatros", font_size="0.8em", color="gray"),
                    rx.text(GolfState.scoring_stats["albatross"], font_size="1.8em", font_weight="bold",
                            color="#8B00FF"),
                    align="center", spacing="0", padding="0.8em", border_radius="md",
                    background_color="rgba(139, 0, 255, 0.1)", width="100%", border="1px solid rgba(139, 0, 255, 0.3)"
                ),
                # Eagle
                rx.vstack(
                    rx.text("Eagle", font_size="0.8em", color="gray"),
                    rx.text(GolfState.scoring_stats["eagle"], font_size="1.8em", font_weight="bold", color="#FF8C00"),
                    align="center", spacing="0", padding="0.8em", border_radius="md",
                    background_color="rgba(255, 140, 0, 0.1)", width="100%", border="1px solid rgba(255, 140, 0, 0.3)"
                ),
                # Birdie
                rx.vstack(
                    rx.text("Birdie", font_size="0.8em", color="gray"),
                    rx.text(GolfState.scoring_stats["birdie"], font_size="1.8em", font_weight="bold", color="#FFD700"),
                    align="center", spacing="0", padding="0.8em", border_radius="md",
                    background_color="rgba(255, 215, 0, 0.1)", width="100%", border="1px solid rgba(255, 215, 0, 0.3)"
                ),
                # Ace
                rx.vstack(
                    rx.text("Ace", font_size="0.8em", color="gray"),
                    rx.text(GolfState.scoring_stats["ace"], font_size="1.8em", font_weight="bold", color="#FF0000"),
                    align="center", spacing="0", padding="0.8em", border_radius="md",
                    background_color="rgba(255, 0, 0, 0.1)", width="100%", border="1px solid rgba(255, 0, 0, 0.3)"
                ),
                width="100%",
                spacing="2",
                justify="between"
            ),
            rx.hstack(
                rx.text("Gesamt: ", font_size="0.9em", color="gray"),
                rx.text(GolfState.scoring_total, font_size="0.9em", font_weight="bold", color="black"),
                rx.spacer(),
                rx.text("Ø pro Runde: ", font_size="0.9em", color="gray"),
                rx.text(GolfState.scoring_average, font_size="0.9em", font_weight="bold", color="black"),
                width="100%",
                justify="between",
                padding_top="0.5em"
            ),
            width="100%",
            padding="1em",
            background_color="#fcfcfc",
            border_radius="lg",
            border="1px solid #e0e0e0",
            box_shadow="sm"
        ),
        width="100%"
    )


def tournament_summary_card():
    """NEW: Displays tournament totals"""
    return rx.box(
        rx.vstack(
            rx.heading("Turnier Übersicht", size="3"),
            rx.grid(
                # Total Tournaments
                rx.vstack(
                    rx.text("Gesamt", font_size="0.8em", color="gray"),
                    rx.text(GolfState.tournament_counts["total"], font_size="1.5em", font_weight="bold"),
                    align="center", spacing="0", padding="0.5em", border_radius="md", background_color="#f0f0f0",
                    width="100%"
                ),
                # With Result
                rx.vstack(
                    rx.text("Mit Ergebnis", font_size="0.8em", color="gray"),
                    rx.text(GolfState.tournament_counts["with_result"], font_size="1.5em", font_weight="bold",
                            color="green"),
                    align="center", spacing="0", padding="0.5em", border_radius="md",
                    background_color="rgba(0, 255, 0, 0.05)", width="100%"
                ),
                # Einzel
                rx.vstack(
                    rx.text("Einzel", font_size="0.8em", color="gray"),
                    rx.text(GolfState.tournament_counts["single"], font_size="1.5em", font_weight="bold", color="blue"),
                    align="center", spacing="0", padding="0.5em", border_radius="md",
                    background_color="rgba(0, 0, 255, 0.05)", width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%",
            ),
            width="100%",
            padding="1em",
            background_color="#fcfcfc",
            border_radius="lg",
            border="1px solid #e0e0e0",
            box_shadow="sm"
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
            rx.hstack(
                rx.heading("🏌️ Golf Performance", size="6"),
                rx.spacer(),
                rx.button("Abmelden", on_click=GolfState.logout, size="2", variant="soft"),
                width="100%",
                padding_y="0.4em",
            ),

            # Player and year selection
            rx.vstack(
                rx.hstack(
                    rx.text("Spieler:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(
                        GolfState.player_options,
                        value=GolfState.selected_player,
                        on_change=GolfState.set_selected_player,
                        placeholder="Spieler auswählen...",
                        width="100%",
                        max_width="400px",
                    ),
                    justify="start", width="100%",
                ),
                rx.hstack(
                    rx.text("Jahr:", font_size="0.9em", font_weight="bold", width="80px"),
                    rx.select(
                        GolfState.year_options,
                        value=GolfState.selected_year,
                        on_change=GolfState.set_selected_year,
                        placeholder="Jahr auswählen...",
                        width="100%",
                        max_width="400px",
                    ),
                    justify="start", width="100%",
                ),
                width="100%",
                spacing="2",
                padding_bottom="1em",
            ),

            rx.hstack(
                rx.hstack(rx.checkbox(checked=GolfState.filter_brutto, on_change=GolfState.toggle_brutto),
                          rx.text("nur Turniere mit Ergebnis", font_size="0.9em"), align="center", spacing="1"),
                rx.hstack(rx.checkbox(checked=GolfState.filter_einzel, on_change=GolfState.toggle_einzel),
                          rx.text("nur Einzel", font_size="0.9em"), align="center", spacing="1"),
                justify="center",
                width="100%",
                spacing="4",
                padding_bottom="1em"
            ),

            # Chart Container
            rx.box(
                rx.cond(
                    GolfState.filtered_df.empty,
                    rx.text("Keine Daten für die gewählten Filter", color="gray", padding="40px", text_align="center",
                            width="100%"),
                    rx.plotly(
                        data=GolfState.figure,
                        use_resize_handler=True,  # Critical for PC/Mobile width
                        style={"width": "100%", "height": "100%"},  # Critical for PC/Mobile width
                        config={
                            "responsive": True,
                            "scrollZoom": True,  # Critical for pinch-to-zoom
                            "displayModeBar": False,
                            "doubleClick": "reset",
                        }
                    )
                ),
                width="100%",
                min_height="450px",
                height="55vh",
                max_height="600px",
                border_radius="md",
                border="1px solid #e0e0e0",
                background_color="white"
            ),

            rx.box(
                rx.text("📊 Mausrad oder Pinch-Geste zum Zoomen | Klicken und Ziehen zum Verschieben",
                        font_size="0.75em", color="gray", text_align="center"),
                width="100%",
                padding="0.5em",
                background_color="#f8f9fa",
                border_radius="md",
            ),

            # Info Cards
            scoring_stats_card(),
            tournament_summary_card(),  # NEW Summary Card

            # Rounds List
            rx.vstack(
                rx.heading("Letzte 5 Runden", size="3"),
                rx.cond(
                    GolfState.last_five_rounds.length == 0,
                    rx.text("Keine Runden für die gewählten Filter", color="gray", padding="20px", text_align="center",
                            width="100%"),
                    rx.foreach(GolfState.last_five_rounds, round_card)
                ),
                width="100%",
                background_color="#fcfcfc",
                padding="1.2em",
                border_radius="lg",
                border="1px solid #e0e0e0",
                box_shadow="sm"
            ),
            width="100%",
            spacing="3",
            padding_x="8px",
            max_width="1400px",
            margin_x="auto"
        ),
        width="100%",
        max_width="100%",
        overflow_x="hidden",
        padding="1.5em",
        background="linear-gradient(180deg, #f8fafc 0%, #ffffff 100%)",
    )


def index() -> rx.Component:
    return rx.cond(
        GolfState.is_authenticated,
        dashboard(),
        login_form()
    )


app = rx.App()
app.add_page(index)