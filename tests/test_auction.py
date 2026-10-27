"""
Unit tests for Auction and Voting Mechanisms.
"""

import pytest
from src.mechanisms.auction import (
    VickreyAuction,
    FirstPriceAuction,
    MultiVehicleAuction,
    AuctionBid,
)
from src.mechanisms.voting import (
    PriorityVotingMechanism,
    VehicleVoter,
)


def test_vickrey_auction_properties():
    """Test Vickrey second-price auction selects highest bidder at second-highest price."""
    auction = VickreyAuction()
    bids = [
        AuctionBid(vehicle_id="V1", true_value=8.0, bid=8.0),
        AuctionBid(vehicle_id="V2", true_value=12.0, bid=12.0),
        AuctionBid(vehicle_id="V3", true_value=5.0, bid=5.0),
    ]

    result = auction.run(bids)

    # Winner should be V2
    assert result.winner == "V2"
    # Payment should be second-highest bid (8.0)
    assert result.payment == 8.0
    # Winner's surplus = 12.0 - 8.0 = 4.0
    assert result.winner_surplus == 4.0
    # Efficient allocation (highest true value won)
    assert result.is_efficient is True
    # Priority ordering
    assert result.ordering == ["V2", "V1", "V3"]


def test_vickrey_emergency_override():
    """Test emergency vehicles get priority override."""
    auction = VickreyAuction()
    bids = [
        AuctionBid(vehicle_id="Normal_1", true_value=10.0, bid=10.0),
        AuctionBid(vehicle_id="Ambulance", true_value=7.0, bid=7.0, is_emergency=True),
    ]

    result = auction.run(bids)
    assert result.winner == "Ambulance"


def test_first_price_auction():
    """Test first-price auction collects winning bid as payment."""
    auction = FirstPriceAuction()
    bids = [
        AuctionBid(vehicle_id="A", true_value=10.0, bid=7.5),
        AuctionBid(vehicle_id="B", true_value=8.0, bid=6.0),
    ]

    result = auction.run(bids)
    assert result.winner == "A"
    assert result.payment == 7.5


def test_priority_voting_mechanism():
    """Test Borda count voting with emergency weight."""
    candidates = ["Car_North", "Car_East", "Ambulance_South"]
    voters = [
        VehicleVoter("Car_North", "north", preferences=["Car_North", "Car_East", "Ambulance_South"]),
        VehicleVoter("Car_East", "east", preferences=["Car_East", "Ambulance_South", "Car_North"]),
        VehicleVoter("Ambulance_South", "south", preferences=["Ambulance_South", "Car_North", "Car_East"], is_emergency=True),
    ]
    for v in voters:
        v.compute_priority_weight()

    result = PriorityVotingMechanism.borda_count(voters, candidates, weighted=True)
    assert result.winner == "Ambulance_South"
