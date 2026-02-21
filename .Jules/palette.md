## 2024-03-22 - [Mobile UX] Chart Interaction
**Learning:** On mobile devices, `scrollZoom: True` in Plotly charts hijacks the page scroll, frustrating users. It can also lead to accidental "reverting" or resizing of the X-axis when users try to scroll past the chart.
**Action:** Disable `scrollZoom` (`config={"scrollZoom": False}`) for full-width charts on mobile-heavy interfaces. Rely on panning (`dragmode="pan"`) for horizontal navigation within the chart, and let the page handle vertical scrolling.
