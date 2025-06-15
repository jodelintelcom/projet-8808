# vision_dashboard.py

import pandas as pd
import plotly.graph_objects as go

# === 1. Load and Prepare Data ===
data_path = "D:/Poly/Project/data/2025_LoL_esports_match_data_from_OraclesElixir.csv"
df = pd.read_csv(data_path, dtype={"url": str})

# Filter for complete games and support role
df = df[(df["datacompleteness"] == "complete") & (df["position"] == "sup")]

# Drop rows with missing values for key metrics
vision_cols = ["wardsplaced", "wardskilled", "visionscore", "gamelength", "result"]
df = df.dropna(subset=vision_cols)

# Convert game length to minutes
df["gamelength_minutes"] = df["gamelength"] / 60

# Compute vision metrics
df["WPM"] = df["wardsplaced"] / df["gamelength_minutes"]
df["WCPM"] = df["wardskilled"] / df["gamelength_minutes"]
df["VSPM"] = df["visionscore"] / df["gamelength_minutes"]
df["VisionEfficiency"] = df["VSPM"] / (df["WPM"] + df["WCPM"])

# Win column (1 = win, 0 = loss)
df["win"] = df["result"]

# === 2. Prepare Dropdown Filter for Players ===

# Unique support players sorted
players = sorted(df["playername"].unique())

# Create visibility masks for each player
visibility_masks = []
for player in players:
    visibility_masks.append(df["playername"] == player)

# Buttons for each player
buttons = []

for i, player in enumerate(players):
    vis = [False] * len(df)
    for idx, val in enumerate(df["playername"] == player):
        vis[idx] = val

    buttons.append(dict(
        label=player,
        method="update",
        args=[{"visible": vis},
              {"title": f"Vision Efficiency Scatter Plot for Support: {player}"}]
    ))

# Button for all supports (all visible)
buttons.insert(0, dict(
    label="All Supports",
    method="update",
    args=[{"visible": [True] * len(df)},
          {"title": "Vision Efficiency Scatter Plot (Supports Only)"}]
))

# === 3. Create Scatter Plot ===

fig = go.Figure()

# Add one trace per data point for full visibility control
for idx, row in df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["WCPM"]],
        y=[row["WPM"]],
        mode='markers',
        marker=dict(
            size=10,  # fixed bubble size
            color='green' if row["win"] == 1 else 'red',
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name=row["playername"],
        hovertemplate=(
            f"Player: {row['playername']}<br>"
            f"Champion: {row['champion']}<br>"
            f"Team: {row['teamname']}<br>"
            f"Wards Placed/min: {row['WPM']:.2f}<br>"
            f"Wards Cleared/min: {row['WCPM']:.2f}<br>"
            f"Vision Score/min: {row['VSPM']:.2f}<br>"
            f"Win: {'Yes' if row['win'] == 1 else 'No'}<extra></extra>"
        ),
        visible=True
    ))

# === 4. Layout and Dropdown Menu ===

fig.update_layout(
    template="plotly_white",
    title="Vision Efficiency Scatter Plot (Supports Only)",
    xaxis_title="Wards Cleared / Min (WCPM)",
    yaxis_title="Wards Placed / Min (WPM)",
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=1.15,
        y=1,
        xanchor="left",
        yanchor="top",
        showactive=True,
        font=dict(size=12),
        bgcolor='LightGray',
        bordercolor='Gray',
        borderwidth=1
    )],
    showlegend=False,
    margin=dict(t=80, r=200)  # room for dropdown on right
)

# === 4b. Add average lines ===

# Compute averages
avg_wpm = df["WPM"].mean()
avg_wcpm = df["WCPM"].mean()

# Add horizontal line for average WPM
fig.add_shape(
    type="line",
    x0=df["WCPM"].min(),
    x1=df["WCPM"].max(),
    y0=avg_wpm,
    y1=avg_wpm,
    line=dict(color="Blue", width=2, dash="dash"),
    name="Avg WPM"
)

# Add vertical line for average WCPM
fig.add_shape(
    type="line",
    x0=avg_wcpm,
    x1=avg_wcpm,
    y0=df["WPM"].min(),
    y1=df["WPM"].max(),
    line=dict(color="Orange", width=2, dash="dash"),
    name="Avg WCPM"
)

# Add annotations for average lines
fig.add_annotation(
    x=df["WCPM"].max(),
    y=avg_wpm,
    text=f"Avg WPM: {avg_wpm:.2f}",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(color="Blue")
)

fig.add_annotation(
    x=avg_wcpm,
    y=df["WPM"].max(),
    text=f"Avg WCPM: {avg_wcpm:.2f}",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(color="Orange")
)



# === 5. Show Plot ===
fig.show()
# Optionally save to HTML:
# fig.write_html("vision_dashboard_output.html", auto_open=True)
