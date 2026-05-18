import numpy as np
from valuation import ROSTER_SLOTS, TOTAL_ROSTER_LIMIT, find_open_roster_slot

class OttoneuAgent:
    def __init__(self, manager_id, budget=400):
        self.manager_id = manager_id
        self.budget = budget
        self.roster = []
        self.roster_counts = {pos: 0 for pos in ROSTER_SLOTS.keys()}
        
        self.alpha = 2 
        self.beta = 2  

    def has_open_slot(self, player_positions):
        # THE FIX: The 40-Man Hard Cap MUST be checked first!
        total_players = sum(self.roster_counts.values())
        if total_players >= TOTAL_ROSTER_LIMIT:
            return False # Immediate rejection
            
        return find_open_roster_slot(self.roster_counts, player_positions) is not None

    def get_limit_price(self, player_value, market_inflation):
        confidence_sample = np.random.beta(self.alpha, self.beta)
        risk_modifier = 0.85 + (confidence_sample * 0.3) 
        
        limit_bid = player_value * market_inflation * risk_modifier
        
        spots_left = TOTAL_ROSTER_LIMIT - len(self.roster)
        max_allowable_bid = self.budget - (spots_left - 1)
        
        return min(int(limit_bid), max_allowable_bid)

    def learn_from_result(self, reward):
        if reward > 0:
            self.alpha += 1 
        else:
            self.beta += 1
