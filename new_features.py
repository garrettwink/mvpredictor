import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from basketball_reference_web_scraper import client

    return (client,)


@app.cell
def _(client):
    player_adv = client.players_advanced_season_totals(season_end_year=2026)
    print(player_adv[9])
    return


if __name__ == "__main__":
    app.run()
