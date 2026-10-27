"""
Auction Mechanisms for intersection right-of-way allocation.

Implements Vickrey (second-price sealed-bid) and first-price auctions
as mechanism design solutions for multi-vehicle intersection management.
Each vehicle bids its urgency (value of going first), and the mechanism
allocates right-of-way efficiently.

Key properties of Vickrey auction:
- Incentive Compatible: truth-telling is a dominant strategy
- Individually Rational: winner's surplus >= 0
- Efficient: highest-value vehicle goes first
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class AuctionBid:
    """
    A bid in the right-of-way auction.

    Attributes
    ----------
    vehicle_id : str
        Identifier for the vehicle.
    true_value : float
        The vehicle's true urgency value (private information).
    bid : float
        The submitted bid amount.
    direction : str
        Approach direction (north, south, east, west).
    is_emergency : bool
        Whether this is an emergency vehicle.
    """
    vehicle_id: str
    true_value: float
    bid: float
    direction: str = ""
    is_emergency: bool = False


@dataclass
class AuctionResult:
    """
    Result of running an auction.

    Attributes
    ----------
    winner : str
        Vehicle ID of the winner.
    payment : float
        Amount the winner pays.
    ordering : list[str]
        Full priority ordering of all vehicles.
    winner_surplus : float
        Winner's net benefit (true_value - payment).
    is_efficient : bool
        Whether the highest-value vehicle won.
    revenue : float
        Total revenue collected by the mechanism.
    details : list[dict]
        Per-vehicle details.
    """
    winner: str
    payment: float
    ordering: list[str]
    winner_surplus: float
    is_efficient: bool
    revenue: float
    details: list[dict] = field(default_factory=list)


class VickreyAuction:
    """
    Vickrey (Second-Price Sealed-Bid) Auction for right-of-way.

    In a Vickrey auction:
    - Each vehicle submits a sealed bid representing their urgency
    - The highest bidder wins and gets right-of-way first
    - The winner pays the SECOND-highest bid
    - Truth-telling (bid = true value) is a dominant strategy

    Examples
    --------
    >>> auction = VickreyAuction()
    >>> bids = [
    ...     AuctionBid("AV1", true_value=8.0, bid=8.0, direction="north"),
    ...     AuctionBid("AV2", true_value=5.0, bid=5.0, direction="east"),
    ...     AuctionBid("AV3", true_value=12.0, bid=12.0, direction="south"),
    ...     AuctionBid("AV4", true_value=3.0, bid=3.0, direction="west"),
    ... ]
    >>> result = auction.run(bids)
    >>> print(f"Winner: {result.winner}, Payment: {result.payment}")
    Winner: AV3, Payment: 8.0
    """

    def run(self, bids: list[AuctionBid]) -> AuctionResult:
        """
        Execute the Vickrey auction.

        Parameters
        ----------
        bids : list[AuctionBid]
            List of bids from participating vehicles.

        Returns
        -------
        AuctionResult
            Auction outcome.
        """
        if not bids:
            raise ValueError("At least one bid is required.")

        if len(bids) == 1:
            return AuctionResult(
                winner=bids[0].vehicle_id,
                payment=0.0,
                ordering=[bids[0].vehicle_id],
                winner_surplus=bids[0].true_value,
                is_efficient=True,
                revenue=0.0,
                details=[{
                    "vehicle_id": bids[0].vehicle_id,
                    "bid": bids[0].bid,
                    "true_value": bids[0].true_value,
                    "position": 1,
                    "payment": 0.0,
                }],
            )

        # Sort by emergency priority, then by bid (descending)
        sorted_bids = sorted(
            bids,
            key=lambda b: (1 if b.is_emergency else 0, b.bid),
            reverse=True,
        )

        winner = sorted_bids[0]
        second_price = sorted_bids[1].bid

        # Full ordering
        ordering = [b.vehicle_id for b in sorted_bids]


        # Check efficiency: did the highest true-value vehicle win?
        highest_value_vehicle = max(bids, key=lambda b: b.true_value)
        is_efficient = winner.vehicle_id == highest_value_vehicle.vehicle_id

        # Build details
        details = []
        for pos, b in enumerate(sorted_bids):
            payment = second_price if pos == 0 else 0.0
            details.append({
                "vehicle_id": b.vehicle_id,
                "bid": b.bid,
                "true_value": b.true_value,
                "direction": b.direction,
                "position": pos + 1,
                "payment": payment,
                "surplus": b.true_value - payment if pos == 0 else 0.0,
            })

        return AuctionResult(
            winner=winner.vehicle_id,
            payment=second_price,
            ordering=ordering,
            winner_surplus=winner.true_value - second_price,
            is_efficient=is_efficient,
            revenue=second_price,
            details=details,
        )

    def prove_incentive_compatibility(
        self, true_value: float, n_opponents: int = 3, n_trials: int = 10000,
        seed: int = 42,
    ) -> dict:
        """
        Empirically demonstrate that truth-telling is optimal in Vickrey auctions.

        Simulates many auction rounds comparing:
        1. Truth-telling (bid = true value)
        2. Overbidding (bid > true value)
        3. Underbidding (bid < true value)

        Parameters
        ----------
        true_value : float
            The vehicle's true urgency.
        n_opponents : int
            Number of opponent vehicles.
        n_trials : int
            Number of simulation trials.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Average surplus for each strategy, proving truth-telling dominates.
        """
        rng = np.random.default_rng(seed)

        strategies = {
            "truth_telling": true_value,
            "overbid_20%": true_value * 1.2,
            "overbid_50%": true_value * 1.5,
            "underbid_20%": true_value * 0.8,
            "underbid_50%": true_value * 0.5,
        }

        results = {name: [] for name in strategies}

        for _ in range(n_trials):
            # Generate random opponent bids
            opponent_values = rng.uniform(0, 20, n_opponents)

            for strat_name, bid_amount in strategies.items():
                all_bids = [
                    AuctionBid("ego", true_value=true_value, bid=bid_amount)
                ]
                for i, v in enumerate(opponent_values):
                    all_bids.append(
                        AuctionBid(f"opp_{i}", true_value=v, bid=v)
                    )

                result = self.run(all_bids)

                if result.winner == "ego":
                    surplus = true_value - result.payment
                else:
                    surplus = 0.0

                results[strat_name].append(surplus)

        return {
            name: {
                "avg_surplus": np.mean(vals),
                "std_surplus": np.std(vals),
                "win_rate": np.mean([1 if v > 0 else 0 for v in vals]),
            }
            for name, vals in results.items()
        }


class FirstPriceAuction:
    """
    First-Price Sealed-Bid Auction (for comparison with Vickrey).

    In a first-price auction:
    - Highest bidder wins
    - Winner pays their OWN bid
    - NOT incentive compatible: bidders shade bids below true value

    Used to demonstrate why Vickrey is preferred.
    """

    def run(self, bids: list[AuctionBid]) -> AuctionResult:
        """Execute the first-price auction."""
        if not bids:
            raise ValueError("At least one bid is required.")

        sorted_bids = sorted(bids, key=lambda b: b.bid, reverse=True)
        winner = sorted_bids[0]

        ordering = [b.vehicle_id for b in sorted_bids]

        highest_value_vehicle = max(bids, key=lambda b: b.true_value)
        is_efficient = winner.vehicle_id == highest_value_vehicle.vehicle_id

        details = []
        for pos, b in enumerate(sorted_bids):
            payment = b.bid if pos == 0 else 0.0
            details.append({
                "vehicle_id": b.vehicle_id,
                "bid": b.bid,
                "true_value": b.true_value,
                "direction": b.direction,
                "position": pos + 1,
                "payment": payment,
                "surplus": b.true_value - payment if pos == 0 else 0.0,
            })

        return AuctionResult(
            winner=winner.vehicle_id,
            payment=winner.bid,  # First price: pay own bid
            ordering=ordering,
            winner_surplus=winner.true_value - winner.bid,
            is_efficient=is_efficient,
            revenue=winner.bid,
            details=details,
        )


def compare_auction_mechanisms(
    n_vehicles: int = 4,
    n_trials: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compare Vickrey and First-Price auctions across many random instances.

    Parameters
    ----------
    n_vehicles : int
        Number of vehicles per auction.
    n_trials : int
        Number of auction instances.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Comparison metrics for both auction types.
    """
    rng = np.random.default_rng(seed)
    vickrey = VickreyAuction()
    first_price = FirstPriceAuction()

    results = {
        "vickrey": {"efficiency": [], "revenue": [], "winner_surplus": []},
        "first_price": {"efficiency": [], "revenue": [], "winner_surplus": []},
    }

    for _ in range(n_trials):
        values = rng.uniform(1, 20, n_vehicles)

        # Vickrey: everyone bids truthfully
        vickrey_bids = [
            AuctionBid(f"V{i}", true_value=v, bid=v)
            for i, v in enumerate(values)
        ]
        v_result = vickrey.run(vickrey_bids)
        results["vickrey"]["efficiency"].append(v_result.is_efficient)
        results["vickrey"]["revenue"].append(v_result.revenue)
        results["vickrey"]["winner_surplus"].append(v_result.winner_surplus)

        # First-price: everyone shades by ~50% of true value (BNE strategy)
        fp_bids = [
            AuctionBid(f"V{i}", true_value=v, bid=v * (n_vehicles - 1) / n_vehicles)
            for i, v in enumerate(values)
        ]
        fp_result = first_price.run(fp_bids)
        results["first_price"]["efficiency"].append(fp_result.is_efficient)
        results["first_price"]["revenue"].append(fp_result.revenue)
        results["first_price"]["winner_surplus"].append(fp_result.winner_surplus)

    return {
        mech: {
            "avg_efficiency": np.mean(data["efficiency"]),
            "avg_revenue": np.mean(data["revenue"]),
            "avg_winner_surplus": np.mean(data["winner_surplus"]),
        }
        for mech, data in results.items()
    }


# Alias for 4+ vehicle right-of-way auction
MultiVehicleAuction = VickreyAuction
