import argparse

from mlb_stock import build_mlb_stock, print_board as print_mlb_board
from prospect_updates import build_prospect_updates
from value_engine import build_player_value_table


def print_prospect_board(updates, limit=25, mode="risers"):
    sort_by = {
        "risers": ("Prospect_Update", False),
        "fallers": ("Prospect_Update", True),
        "prospects": ("Updated_Prospect_Value", False),
    }[mode]
    cols = [
        "Name", "Org", "Savant_Level", "Player_Type", "Pos", "Pipeline_Rank",
        "Prior_Value", "Prospect_Update", "Updated_Prospect_Value", "Confidence_Label",
    ]
    output = updates.sort_values(sort_by[0], ascending=sort_by[1]).head(limit)
    print(output[cols].to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def print_roster_board(values, limit=25, mode="underpriced"):
    sort_map = {
        "underpriced": ("Current_Surplus", False),
        "buy-low": ("Stock_Change", True),
        "sell-high": ("Stock_Change", False),
    }
    sort_by, ascending = sort_map[mode]
    candidates = values.copy()
    if mode == "buy-low":
        candidates = candidates[candidates["Current_Surplus"] >= -5]
    if mode == "sell-high":
        candidates = candidates[candidates["Current_Surplus"] >= 0]
    cols = [
        "Team Name", "Name", "Positions", "Salary", "Current_Value", "Current_Surplus",
        "Stock_Change", "Value_Source", "Confidence_Label",
    ]
    print(
        candidates.sort_values(sort_by, ascending=ascending)
        .head(limit)[cols]
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


def main():
    parser = argparse.ArgumentParser(description="Rolling Ottoneu stock board.")
    parser.add_argument(
        "command",
        nargs="?",
        default="risers",
        choices=[
            "risers", "fallers", "prospects", "prospect-risers", "prospect-fallers",
            "underpriced", "buy-low", "sell-high",
        ],
    )
    parser.add_argument("limit", nargs="?", type=int, default=25)
    parser.add_argument("--rosters", default="current_rosters.csv")
    parser.add_argument("--hitters", default="hitters_ros.csv")
    parser.add_argument("--pitchers", default="pitchers_ros.csv")
    parser.add_argument("--avg", default="fgpts_avgvalues.csv")
    parser.add_argument("--prospects", default="data/Baseball Composite Prospect List 2026 - List.csv")
    parser.add_argument("--prospect-updates", default="prospect_value_updates.csv")
    args = parser.parse_args()

    if args.command in {"risers", "fallers"}:
        stock = build_mlb_stock()
        print_mlb_board(stock, mode=args.command, limit=args.limit)
        return

    if args.command in {"prospects", "prospect-risers", "prospect-fallers"}:
        updates = build_prospect_updates(avg_csv=args.avg)
        prospect_mode = {
            "prospects": "prospects",
            "prospect-risers": "risers",
            "prospect-fallers": "fallers",
        }[args.command]
        print_prospect_board(updates, limit=args.limit, mode=prospect_mode)
        return

    values, _ = build_player_value_table(
        args.rosters, args.hitters, args.pitchers, args.avg, args.prospects, args.prospect_updates
    )
    print_roster_board(values, limit=args.limit, mode=args.command)


if __name__ == "__main__":
    main()
