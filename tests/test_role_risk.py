import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from role_risk import apply_role_risk_adjustment, role_risk_adjustment


def test_role_risk_adjustment_haircuts_low_confidence_rp_to_sp():
    row = {
        "Role_Change": "RP_TO_SP",
        "ROS_Value": 36.0,
        "ROS_IP": 110,
        "YTD_ROS_Gap": -70,
        "YTD_Value": -40,
        "Confidence_Label": "Low",
    }

    assert role_risk_adjustment(row) == -14.4


def test_role_risk_adjustment_ignores_stable_roles():
    row = {"Role_Change": "STABLE_SP", "ROS_Value": 36.0, "Confidence_Label": "Low"}

    assert role_risk_adjustment(row) == 0.0


def test_apply_role_risk_adjustment_updates_value_column():
    df = pd.DataFrame([{
        "Role_Change": "RP_TO_SP",
        "ROS_Value": 20.0,
        "ROS_IP": 85,
        "YTD_ROS_Gap": -25,
        "YTD_Value": -5,
        "Confidence_Label": "Medium",
        "Current_Value": 20.0,
    }])

    out = apply_role_risk_adjustment(df, "Current_Value")

    assert out.loc[0, "Role_Risk_Adjustment"] < 0
    assert out.loc[0, "Current_Value"] < 20.0
