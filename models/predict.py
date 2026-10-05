from dataset import X_test, y_test, id_test
from train import model

ranking_score = model.predict(X_test)

results = id_test.copy()
results["actual_vote_share"] = y_test.to_numpy()
results["ranking_score"] = ranking_score

results["prediction_rank"] = (
    results.groupby("season")["ranking_score"]
    .rank(method="first", ascending=False)
    .astype(int)
)
results["actual_rank"] = (
    results.groupby("season")["actual_vote_share"]
    .rank(method="min", ascending=False)
    .astype(int)
)

for season, season_results in results.groupby("season"):
    print(f"\nSeason: {season}")
    print(
        season_results
        .sort_values("prediction_rank")
        .head(10)
        [["prediction_rank", "player_name", "ranking_score", "actual_vote_share", "actual_rank"]]
        .to_string(index=False)
    )

    winner_hits = (results.loc[results["prediction_rank"] == 1, "actual_rank"] == 1).sum()
    top_three_hits = (results.loc[results["prediction_rank"] <= 3, "actual_rank"] == 1).sum()
    season_count = results["season"].nunique()

print(f"\nWinner hit rate: {winner_hits}/{season_count} seasons")
print(f"Top-3 winner coverage: {top_three_hits}/{season_count} seasons")