import pandas as pd


def _number(value, default=0.0):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return default
    return float(converted)


def role_risk_adjustment(row):
    role = str(row.get("Role_Change", "") or "").upper()
    if role != "RP_TO_SP":
        return 0.0

    ros_value = _number(row.get("ROS_Value"))
    ros_ip = _number(row.get("ROS_IP"))
    ytd_gap = _number(row.get("YTD_ROS_Gap"))
    ytd_value = _number(row.get("YTD_Value"))
    confidence = str(row.get("Confidence_Label", "") or "").strip().lower()

    penalty_rate = 0.18
    if ros_ip < 120:
        penalty_rate += 0.08
    if ros_ip < 90:
        penalty_rate += 0.04
    if confidence == "low":
        penalty_rate += 0.07
    elif confidence == "medium":
        penalty_rate += 0.04
    if ytd_gap <= -20:
        penalty_rate += 0.05
    if ytd_value < 0:
        penalty_rate += 0.03

    penalty = ros_value * min(penalty_rate, 0.40)
    return -round(penalty, 3)


def apply_role_risk_adjustment(df, value_col):
    out = df.copy()
    if value_col not in out.columns:
        out["Role_Risk_Adjustment"] = 0.0
        return out
    out["Role_Risk_Adjustment"] = out.apply(role_risk_adjustment, axis=1)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0) + out["Role_Risk_Adjustment"]
    return out
