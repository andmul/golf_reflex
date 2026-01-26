import reflex as rx
import pandas as pd
import plotly.graph_objects as go
import os


# ==============================================================================
# Load data ONCE
# ==============================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
print(DATA_PATH)
df_base = pd.read_csv(DATA_PATH,index_col=0)
df_base = df_base.reset_index(drop=True)
df_base["Datum"] = pd.to_datetime(df_base["Datum"], errors="coerce")
df_base = df_base.dropna(subset=["Datum"])

print(df_base.columns)
print("Loaded rows:", len(df_base))
print(df_base.head())



# ==============================================================================
# Helpers
# ==============================================================================
def build_hovertext(df):
    return df.apply(
        lambda r: (
            f"<b>Date:</b> {r['Datum'].strftime('%d-%m-%y')}<br>"
            f"<b>Turnier:</b> {r.get('Turnier', '–')}<br>"
            f"<b>Club:</b> {r.get('Club', '–')}<br>"
            f"<b>Spielmodus:</b> {r.get('Spielmodus', '–')}<br>"
            f"<b>HCP:</b> {r.get('HCP', 0)}<br>"
            f"<b>Brutto:</b> {int(r.get('Brutto', 0))}<br>"
            f"<b>Par:</b> "
            f"{r.get('Par')} / {int(r['uPar']):+d}"
            if not pd.isna(r.get("uPar"))
            else f"{r.get('Par')}"
        ),
        axis=1,
    )


def build_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[10, 15, 7]))
    return fig


# ==============================================================================
# Reflex State
# ==============================================================================
class AppState(rx.State):
    nur_turniere_mit_ergebnis: bool = False
    nur_individuell: bool = False

    @rx.var
    def filtered_df(self) -> pd.DataFrame:
        df = df_base.copy()

        if self.nur_turniere_mit_ergebnis:
            df = df[df["Brutto"] != 0]

        if self.nur_individuell:
            df = df[df["Spielmodus"].str.startswith("Einzel", na=False)]

        return df.reset_index(drop=True)

    @rx.var
    def row_count(self) -> int:
        return len(self.filtered_df)

    @rx.var
    def figure(self) -> go.Figure:
        if self.filtered_df.empty:
            return go.Figure().update_layout(
                title="Keine Daten verfügbar",
                template="plotly_white",
            )
        return build_figure(self.filtered_df)


    


# ==============================================================================
# UI
# ==============================================================================
def index():
    return rx.container(
        rx.heading("🏌️ Turnier-Analyse", size="3"),


        rx.hstack(
            rx.checkbox(
                "nur Turniere mit Ergebnis",
                checked=AppState.nur_turniere_mit_ergebnis,
                on_change=AppState.set_nur_turniere_mit_ergebnis,
            ),
            rx.checkbox(
                "nur individuelle Ergebnisse",
                checked=AppState.nur_individuell,
                on_change=AppState.set_nur_individuell,
            ),
            spacing="5",
            padding="1em",
            background="gray.50",
            border_radius="8px",
        ),
   rx.text(
            AppState.row_count,
            color="gray.600",
        ),

    rx.plotly(AppState.figure),

        max_width="1450px",
        padding="2em",
    )


app = rx.App()
app.add_page(index)

