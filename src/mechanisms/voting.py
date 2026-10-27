"""
Voting and Social Choice Mechanisms for Intersection Coordination.

Implements social choice and voting mechanisms for right-of-way allocation
when monetary transfers (auctions) are undesirable or prohibited (e.g., civic fairness).

Includes:
- Priority-weighted voting (emergency vehicles, high-occupancy vehicles, waiting-time weights)
- Borda count mechanism for intersection passage sequences
- Plurality voting for signal phase selection
- Condorcet winner analysis for order preferences
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class VehicleVoter:
    """
    Vehicle voter representation.

    Attributes
    ----------
    vehicle_id : str
        Unique identifier.
    direction : str
        Approach direction ('north', 'south', 'east', 'west').
    weight : float
        Voting weight based on priority (occupancy, emergency, delay).
    preferences : list[str]
        Ordered preference list over candidate phases or vehicle sequences.
    is_emergency : bool
        True if vehicle is an emergency vehicle (granted veto/maximal weight).
    occupancy : int
        Number of passengers.
    waiting_time : float
        Waiting time in seconds at the stop line.
    """
    vehicle_id: str
    direction: str
    preferences: List[str]
    weight: float = 1.0
    is_emergency: bool = False
    occupancy: int = 1
    waiting_time: float = 0.0

    def compute_priority_weight(
        self,
        emergency_multiplier: float = 10.0,
        occupancy_weight: float = 0.5,
        delay_weight: float = 0.2,
    ) -> float:
        """Compute composite priority voting weight."""
        if self.is_emergency:
            self.weight = emergency_multiplier * (1.0 + self.waiting_time * 0.1)
        else:
            self.weight = 1.0 + (self.occupancy - 1) * occupancy_weight + self.waiting_time * delay_weight
        return self.weight


@dataclass
class VotingResult:
    """Outcome of a voting mechanism."""
    winner: str
    scores: Dict[str, float]
    ranking: List[str]
    mechanism: str
    is_condorcet_winner: bool = False
    details: Dict = field(default_factory=dict)


class PriorityVotingMechanism:
    """
    Implements voting algorithms for intersection right-of-way allocation.
    """

    @staticmethod
    def plurality_vote(
        voters: List[VehicleVoter],
        candidates: List[str],
        weighted: bool = True,
    ) -> VotingResult:
        """
        Plurality voting: each voter casts vote for top preference.
        """
        scores = {c: 0.0 for c in candidates}
        for voter in voters:
            if not voter.preferences:
                continue
            top_choice = voter.preferences[0]
            if top_choice in scores:
                weight = voter.weight if weighted else 1.0
                scores[top_choice] += weight

        sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranking = [c for c, _ in sorted_candidates]
        winner = ranking[0] if ranking else ""

        return VotingResult(
            winner=winner,
            scores=scores,
            ranking=ranking,
            mechanism="Weighted Plurality" if weighted else "Plurality",
            details={"vote_shares": {c: scores[c] / max(1e-9, sum(scores.values())) for c in scores}}
        )

    @staticmethod
    def borda_count(
        voters: List[VehicleVoter],
        candidates: List[str],
        weighted: bool = True,
    ) -> VotingResult:
        """
        Borda Count mechanism:
        For m candidates, top gets m-1 points, second gets m-2, ..., last gets 0.
        Points are scaled by voter weight if weighted=True.
        """
        m = len(candidates)
        scores = {c: 0.0 for c in candidates}

        for voter in voters:
            weight = voter.weight if weighted else 1.0
            # Rank candidates according to preference
            pref_set = set(voter.preferences)
            # Add missing candidates at the end
            full_pref = [c for c in voter.preferences if c in candidates]
            full_pref += [c for c in candidates if c not in pref_set]

            for rank, candidate in enumerate(full_pref):
                points = (m - 1 - rank) * weight
                scores[candidate] += points

        sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranking = [c for c, _ in sorted_candidates]
        winner = ranking[0] if ranking else ""

        # Check Condorcet property
        is_condorcet = PriorityVotingMechanism._check_condorcet(winner, voters, candidates)

        return VotingResult(
            winner=winner,
            scores=scores,
            ranking=ranking,
            mechanism="Weighted Borda Count" if weighted else "Borda Count",
            is_condorcet_winner=is_condorcet,
            details={"borda_points": scores}
        )

    @staticmethod
    def _check_condorcet(
        candidate: str,
        voters: List[VehicleVoter],
        candidates: List[str],
    ) -> bool:
        """Check if candidate beats every other candidate in pairwise head-to-head."""
        if not candidate or len(candidates) <= 1:
            return True

        for other in candidates:
            if other == candidate:
                continue
            cand_votes = 0.0
            other_votes = 0.0
            for v in voters:
                c_idx = v.preferences.index(candidate) if candidate in v.preferences else 999
                o_idx = v.preferences.index(other) if other in v.preferences else 999
                if c_idx < o_idx:
                    cand_votes += v.weight
                elif o_idx < c_idx:
                    other_votes += v.weight
            if cand_votes <= other_votes:
                return False
        return True
