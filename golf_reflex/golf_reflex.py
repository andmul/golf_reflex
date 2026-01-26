
#    return rx.container(
#        rx.heading("Minimal Reflex + Plotly", size="5"),  # Fixed size parameter
#        rx.plotly(data=fig),  # Pass the Figure object directly
#        spacing="4",
#        padding="1em",
#    )
#

import plotly.graph_objects as go
import reflex as rx
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
print(DATA_PATH)
df_base = pd.read_csv(DATA_PATH,index_col=0)
df_base = df_base.reset_index(drop=True)
df_base["Datum"] = pd.to_datetime(df_base["Datum"], errors="coerce")
df_base = df_base.dropna(subset=["Datum"])


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
    required = {"Datum", "HCP"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.sort_values("Datum").reset_index(drop=True)
    df["ContinuousIndex"] = range(len(df))
    df["HoverText"] = build_hovertext(df)

    hcp_min, hcp_max = df["HCP"].min(), df["HCP"].max()
    hcp_pad = max((hcp_max - hcp_min) * 0.15, 1)

    fig = go.Figure()

    # HCP line
    fig.add_trace(go.Scatter(
        x=df["ContinuousIndex"],
        y=df["HCP"],
        mode="lines+markers",
        line=dict(color="#2C7BE5", width=2.5),
        marker=dict(size=7, line=dict(width=1, color="white")),
        text=df["HoverText"],
        hoverinfo="text",
        name="HCP",
    ))

    # Brutto bars
    fig.add_trace(go.Bar(
        x=df["ContinuousIndex"],
        y=df["Brutto"],
        width=0.45,
        marker=dict(color="rgba(80,200,120,0.75)"),
        text=[int(v) if v > 0 else "" for v in df["Brutto"]],
        textposition="inside",
        hovertext=df["HoverText"],
        hoverinfo="text",
        name="Brutto",
        yaxis="y2",
    ))

    # Winter breaks
    gaps = df["Datum"].diff().dt.days > 50
    for i in df.index[gaps]:
        x = (df.loc[i - 1, "ContinuousIndex"] + df.loc[i, "ContinuousIndex"]) / 2
        fig.add_vline(x=x, line_dash="dot", line_color="firebrick", opacity=0.5)
        fig.add_annotation(
            x=x, y=1.06, xref="x", yref="paper",
            text="⛄ Winter", showarrow=False,
            font=dict(size=10, color="firebrick"),
        )

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        bargap=0.2,
        showlegend=False,
        title=dict(text="Turniere & Handicap-Verlauf", x=0.5),
        xaxis=dict(
            tickvals=df["ContinuousIndex"],
            ticktext=df["Datum"].dt.strftime("%d-%m-%y"),
            tickangle=45,
            rangeslider=dict(visible=True, thickness=0.06),
        ),
        yaxis=dict(
            title="HCP",
            range=[hcp_min - hcp_pad, hcp_max + hcp_pad],
        ),
        yaxis2=dict(
            title="Brutto",
            overlaying="y",
            side="right",
        ),
    )

    return fig












class State(rx.State):
    count: int = 0

    def increment(self):
        self.count += 1

    def decrement(self):
        self.count -= 1

# Create minimal plotly chart
figure = go.Figure(
    go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[2, 3, 1, 4, 2],
        mode='lines+markers',
    )
)


def index():
    return rx.hstack(
        rx.button(
            "Decrement",
            color_scheme="ruby",
            on_click=State.decrement,
        ),
        rx.heading(State.count, font_size="2em"),
        rx.button(
            "Increment",
            color_scheme="grass",
            on_click=State.increment,
        ),
        rx.plotly(data=build_figure(df_base)),
        spacing="4",
    )
app = rx.App()
app.add_page(index)