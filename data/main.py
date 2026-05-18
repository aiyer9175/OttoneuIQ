import pandas as pd
from valuation import load_player_pool
from agent import OttoneuAgent
from auction import AuctionEnvironment

def main():
    print("--- Booting Ottoneu Quant Desk v1.0 ---")
    
    # 1. Load the Asset Pool (All 3 CSVs)
    prospect_file = "Baseball Composite Prospect List 2026 - List.csv"
    
    try:
        pool = load_player_pool("batters_auctioncalc.csv", "pitchers_auctioncalc.csv", prospect_file)
        print(f"Loaded {len(pool)} rosterable assets, including Dynasty Prospects!")
    except Exception as e:
        print(f"Error loading files. Check filenames: {e}")
        return

    # 2. Initialize the 12 Agents
    agents = [OttoneuAgent(manager_id=i) for i in range(12)]
    env = AuctionEnvironment(agents, pool)
    
    print("\nExecuting FULL LOB Draft Simulation (Targeting 480 Picks)...")
    draft_history = []
    roster_rows = []
    
    # 3. The Full Marathon Loop
    for pick_num, player in enumerate(pool):
        # The Bouncer: Stop the draft if all 12 teams hit 40 players (480 total)
        total_rostered = sum([len(a.roster) for a in agents])
        if total_rostered >= 480:
            print("\n[LEAGUE FULL] All 480 roster spots have been filled.")
            break
            
        # Run the Limit Order Book transaction
        winner_idx, limit_bid, price, assigned_slot = env.run_lob_auction(player)
        
        if winner_idx is not None:
            winner = f"Team {winner_idx + 1}"
            draft_history.append({
                "Pick": len(draft_history) + 1,
                "Player": player['Name'],
                "Value": round(player['dollars'], 2),
                "Limit_Bid": limit_bid,
                "Price": price,
                "Winner": winner
            })
            roster_rows.append({
                "Team": winner,
                "Player": player['Name'],
                "Slot": assigned_slot,
                "Positions": "/".join(player['positions']),
                "Salary": price,
                "Value": round(player['dollars'], 2),
                "Surplus": round(player['dollars'] - price, 2),
                "Is_Prospect": bool(player.get('is_prospect', False))
            })

    # 4. Generate the Bloomberg Report
    df_results = pd.DataFrame(draft_history)
    df_results.to_csv("full_480_draft_results.csv", index=False)
    df_rosters = pd.DataFrame(roster_rows)
    df_rosters.to_csv("final_team_rosters.csv", index=False)
    
    print("\n--- Simulation Complete ---")
    print(f"Total Assets Cleared: {len(df_results)}")
    
    print("\n--- Post-Draft Audit ---")
    for a in agents:
        roster_size = sum(a.roster_counts.values())
        print(f"Team {a.manager_id + 1}: {roster_size} Players | ${a.budget} Remaining")

    print("\nSuccess! Look for 'full_480_draft_results.csv' and 'final_team_rosters.csv' in your folder.")

if __name__ == "__main__":
    main()
