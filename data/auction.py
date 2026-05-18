from valuation import assign_player_to_roster_slot

class AuctionEnvironment:
    def __init__(self, agents, pool):
        self.agents = agents
        self.pool = pool

    def get_market_inflation(self):
        return 1.15 # Base premium for top players

    def run_lob_auction(self, player):
        limit_orders = []
        inflation = self.get_market_inflation()

        for i, agent in enumerate(self.agents):
            if agent.has_open_slot(player['positions']) and player['dollars'] > 0:
                bid = agent.get_limit_price(player['dollars'], inflation)
                if bid >= 1:
                    limit_orders.append((bid, i))

        if not limit_orders:
            return None, 0, 0, None

        limit_orders.sort(reverse=True, key=lambda x: x[0])
        highest_bid, winner_idx = limit_orders[0]
        
        if len(limit_orders) > 1:
            clearing_price = limit_orders[1][0] + 1
        else:
            clearing_price = 1

        clearing_price = min(clearing_price, highest_bid)
        winner = self.agents[winner_idx]

        assigned_slot = assign_player_to_roster_slot(winner.roster_counts, player['positions'])

        winner.budget -= clearing_price
        winner.roster.append(player['Name'])
        
        reward = (player['dollars'] - clearing_price) + 2.0
        winner.learn_from_result(reward)

        return winner_idx, highest_bid, clearing_price, assigned_slot
