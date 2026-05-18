import os
import sys
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
sys.path.insert(0, DATA_DIR)

from auction_lab import run_auction_sims, run_bandit_training
from data_sources import resolve_data_paths, source_status
from live_auction import (
    LiveAuctionRoom,
    active_lineup_rows,
    max_legal_bid,
    max_legal_bid_for_player,
    roster_size,
    roster_sort_key,
)
from player_trends import build_player_trend_table, resolve_trend_player, trend_waterfall_rows
from post_auction import build_arbitration_report, build_reports, resolve_team_name
from prospect_updates import build_prospect_updates
from roster_audit import build_roster_audit
from snapshots import build_snapshot
from trend_history import build_trend_history, latest_movement, player_history
from trade_evaluator import (
    build_waiver_wire_board,
    evaluate_trade,
    fetch_ottoneu_roster_export,
    parse_players,
    recommend_trade_packages,
)
from valuation import TOTAL_ROSTER_LIMIT, load_player_pool
from value_engine import build_player_value_table, load_or_build_mlb_stock


st.set_page_config(page_title="OttoneuIQ", layout="wide")
st.title("OttoneuIQ")


@st.cache_data(show_spinner=False)
def cached_prospect_updates(data_source, refresh_token):
    paths = resolve_data_paths(data_source)
    return build_prospect_updates(
        pipeline_csv=paths.pipeline,
        composite_csv=paths.prospects,
        avg_csv=paths.avg_values,
        stuff_plus_csv=paths.stuff_plus,
    )


@st.cache_data(show_spinner=False)
def cached_values(data_source, refresh_token, roster_path=None):
    paths = resolve_data_paths(data_source)
    return build_player_value_table(
        rosters=roster_path or paths.rosters,
        hitters=paths.hitters_ros,
        pitchers=paths.pitchers_ros,
        avg=paths.avg_values,
        prospects=paths.prospects,
        mlb_stock=paths.mlb_stock,
        mlb_status=paths.mlb_status,
    )


@st.cache_data(show_spinner=False)
def cached_player_pool(data_source, refresh_token):
    paths = resolve_data_paths(data_source)
    return load_player_pool(
        paths.batters_auction,
        paths.pitchers_auction,
        paths.prospects,
    )


@st.cache_data(show_spinner=False)
def cached_player_trends(data_source, refresh_token, roster_path=None):
    paths = resolve_data_paths(data_source)
    values, _ = cached_values(data_source, refresh_token, roster_path)
    stock = load_or_build_mlb_stock(paths.mlb_stock)
    return build_player_trend_table(values=values, stock=stock)


@st.cache_data(show_spinner=False)
def cached_trend_history(refresh_token):
    return build_trend_history()


def default_data_source():
    return "cache" if source_status().get("latest_cache") else "static"


def selected_data_source():
    if "data_source" not in st.session_state:
        st.session_state.data_source = default_data_source()
    if "data_refresh_token" not in st.session_state:
        st.session_state.data_refresh_token = 0
    return st.session_state.data_source, st.session_state.data_refresh_token


def selected_roster_path(data_source):
    if st.session_state.get("league_roster_mode") == "upload":
        return st.session_state.get("league_roster_upload_path")
    if st.session_state.get("league_roster_mode") == "league_id":
        return st.session_state.get("league_roster_fetch_path")
    return resolve_data_paths(data_source).rosters


def auction_room():
    data_source, refresh_token = selected_data_source()
    if "auction_room" not in st.session_state:
        st.session_state.auction_room = LiveAuctionRoom(
            pool=cached_player_pool(data_source, refresh_token),
            human_team=0,
            bid_seconds=60,
            auto_human=False,
        )
        st.session_state.auction_current = None
    return st.session_state.auction_room


def auction_summary(room):
    return pd.DataFrame([
        {
            "Team": room.team_label(idx),
            "Players": roster_size(agent),
            "Budget": agent.budget,
            "Max Bid": max_legal_bid(agent),
            "You": idx == room.human_team,
        }
        for idx, agent in enumerate(room.agents)
    ])


def team_roster_dataframe(room, team_idx):
    rows = [row for row in room.roster_rows if row["Team"] == room.team_label(team_idx)]
    if not rows:
        return pd.DataFrame(columns=["Slot", "Player", "Salary", "Value", "Surplus", "Positions", "Is_Prospect"])
    rows = sorted(rows, key=roster_sort_key)
    return pd.DataFrame(rows)[["Slot", "Player", "Salary", "Value", "Surplus", "Positions", "Is_Prospect"]]


def team_active_lineup_dataframe(room, team_idx):
    rows = [row for row in room.roster_rows if row["Team"] == room.team_label(team_idx)]
    return pd.DataFrame(active_lineup_rows(rows))


def trend_value_chart(player_rows):
    chart_data = player_rows.copy()
    chart_data["Snapshot_Date"] = pd.to_datetime(chart_data["Snapshot_Date"], errors="coerce")
    chart_data["Snapshot_Label"] = chart_data["Snapshot_Date"].dt.strftime("%Y-%m-%d")
    chart_data["Snapshot_Label"] = chart_data["Snapshot_Label"].fillna(chart_data["Snapshot"].astype(str))
    for col in ["Current_Value", "Context_Value"]:
        chart_data[col] = pd.to_numeric(chart_data[col], errors="coerce")
    chart_data = chart_data.dropna(subset=["Current_Value", "Context_Value"])
    if chart_data.empty:
        return None

    chart_data["Low_Value"] = chart_data[["Current_Value", "Context_Value"]].min(axis=1)
    chart_data["High_Value"] = chart_data[["Current_Value", "Context_Value"]].max(axis=1)
    label_order = chart_data["Snapshot_Label"].tolist()
    x_axis = alt.X(
        "Snapshot_Label:N",
        sort=label_order,
        title="Snapshot Date",
        axis=alt.Axis(labelAngle=0, labelOverlap=False),
    )
    tooltip = [
        alt.Tooltip("Snapshot_Label:N", title="Date"),
        alt.Tooltip("Current_Value:Q", title="ROS $", format=".1f"),
        alt.Tooltip("Context_Value:Q", title="Context $", format=".1f"),
    ]
    base = alt.Chart(chart_data).encode(x=x_axis)
    value_range = base.mark_rule(color="#8a8f98", size=5).encode(
        y=alt.Y("Low_Value:Q", title="Dollars", scale=alt.Scale(zero=False)),
        y2="High_Value:Q",
        tooltip=tooltip,
    )
    ros_tick = base.mark_tick(color="#4c78a8", size=24, thickness=3).encode(
        y=alt.Y("Current_Value:Q", title="Dollars", scale=alt.Scale(zero=False)),
        tooltip=tooltip,
    )
    context_line = base.mark_line(color="#f58518", point=True, strokeWidth=3).encode(
        y=alt.Y("Context_Value:Q", title="Dollars", scale=alt.Scale(zero=False)),
        tooltip=tooltip,
    )
    return (value_range + ros_tick + context_line).properties(height=320)


def start_next_nomination(room, human_choice=None):
    nominator = room.next_nominator()
    if nominator is None:
        return None
    if nominator == room.human_team and human_choice:
        player = next((p for p in room.pool if p["Name"] == human_choice), None)
        if player and room.team_can_roster(nominator, player):
            room.remove_player(player)
        else:
            player = room.simulated_nominate_player(nominator)
    else:
        player = room.simulated_nominate_player(nominator)
    if player is None:
        return None
    current_bid = 1 if room.team_can_roster(nominator, player) else 0
    high_bidder = nominator if current_bid else None
    st.session_state.auction_current = {
        "player": player,
        "nominator": nominator,
        "current_bid": current_bid,
        "high_bidder": high_bidder,
        "passed": set(),
        "human_passed": False,
        "ai_limits": {i: room.ai_limit(i, player) for i in range(12) if i != room.human_team},
        "bid_log": [
            {
                "Bid": current_bid,
                "Team": room.team_label(nominator),
                "Action": "opening nomination bid" if current_bid else "no opening bid",
            }
        ],
    }
    return st.session_state.auction_current


def run_ai_round(room):
    state = st.session_state.auction_current
    if not state:
        return
    while True:
        bidder = room.choose_ai_bidder(
            state["ai_limits"],
            state["current_bid"],
            state["high_bidder"],
            state["passed"],
        )
        if bidder is None:
            break
        state["current_bid"] = room.next_ai_bid(state["ai_limits"][bidder], state["current_bid"])
        state["high_bidder"] = bidder
        state.setdefault("bid_log", []).append({
            "Bid": state["current_bid"],
            "Team": room.team_label(bidder),
            "Action": f"AI bid, limit ${state['ai_limits'][bidder]}",
        })
        if state["high_bidder"] == room.human_team:
            break
        if room.no_live_bidders(
            state["player"],
            state["ai_limits"],
            state["current_bid"],
            state["high_bidder"],
            state["passed"],
            state["human_passed"],
        ):
            break


def complete_if_no_live_bidders(room):
    state = st.session_state.auction_current
    if not state:
        return False
    if room.no_live_bidders(
        state["player"],
        state["ai_limits"],
        state["current_bid"],
        state["high_bidder"],
        state["passed"],
        state["human_passed"],
    ):
        complete_streamlit_auction(room)
        return True
    return False


def complete_streamlit_auction(room):
    state = st.session_state.auction_current
    if not state:
        return
    if state["high_bidder"] is None:
        room.pool.append(state["player"])
    else:
        state.setdefault("bid_log", []).append({
            "Bid": state["current_bid"],
            "Team": room.team_label(state["high_bidder"]),
            "Action": "sold",
        })
        room.complete_purchase(
            state["player"],
            state["high_bidder"],
            state["current_bid"],
            state["nominator"],
            quiet=True,
        )
        room.save_reports()
    st.session_state.auction_current = None


def upload_dir_path():
    upload_dir = os.path.join(tempfile.gettempdir(), "ottoneuiq_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def write_upload(uploaded_file, filename):
    if not uploaded_file:
        return None
    upload_dir = upload_dir_path()
    path = os.path.join(upload_dir, filename)
    with open(path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return path


def fetch_league_rosters(league_id, filename_prefix):
    fetched_path = os.path.join(upload_dir_path(), f"{filename_prefix}_{league_id}_rosterexport.csv")
    fetch_ottoneu_roster_export(league_id, fetched_path)
    return fetched_path


with st.sidebar:
    st.header("Data")
    active_source = st.session_state.get("data_source", default_data_source())
    data_source_choice = st.selectbox("Source", ["static", "cache"], index=0 if active_source == "static" else 1)
    if data_source_choice != active_source:
        st.session_state.data_source = data_source_choice
        st.session_state.data_refresh_token = st.session_state.get("data_refresh_token", 0) + 1
        st.session_state.pop("auction_room", None)
        st.session_state.auction_current = None
        st.rerun()
    if st.button("Refresh Remote Cache"):
        try:
            cache_dir, _ = build_snapshot(source="remote")
            st.session_state.data_source = "cache"
            st.session_state.data_refresh_token = st.session_state.get("data_refresh_token", 0) + 1
            st.session_state.pop("auction_room", None)
            st.session_state.auction_current = None
            cached_values.clear()
            cached_player_pool.clear()
            cached_player_trends.clear()
            cached_trend_history.clear()
            cached_prospect_updates.clear()
            st.success(f"Cached data at {cache_dir}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    status = source_status()
    if status["latest_cache"]:
        st.caption(f"Latest cache: {status['latest_cache']}")
    st.header("League Roster")
    roster_mode_label = st.selectbox(
        "Roster source",
        ["Active data source", "Upload CSV", "Ottoneu league number"],
        index={"active": 0, "upload": 1, "league_id": 2}.get(st.session_state.get("league_roster_mode", "active"), 0),
    )
    roster_mode = {
        "Active data source": "active",
        "Upload CSV": "upload",
        "Ottoneu league number": "league_id",
    }[roster_mode_label]
    if roster_mode != st.session_state.get("league_roster_mode", "active"):
        st.session_state.league_roster_mode = roster_mode
        st.session_state.data_refresh_token = st.session_state.get("data_refresh_token", 0) + 1
        cached_values.clear()
        cached_player_trends.clear()
        st.rerun()
    if roster_mode == "upload":
        league_roster_upload = st.file_uploader("Roster export CSV", type=["csv"], key="global_roster_upload")
        uploaded_path = write_upload(league_roster_upload, "global_league_rosters.csv")
        if uploaded_path and uploaded_path != st.session_state.get("league_roster_upload_path"):
            st.session_state.league_roster_upload_path = uploaded_path
            st.session_state.data_refresh_token = st.session_state.get("data_refresh_token", 0) + 1
            cached_values.clear()
            cached_player_trends.clear()
            st.rerun()
    elif roster_mode == "league_id":
        league_id = st.text_input("League number", value=st.session_state.get("league_id", ""), placeholder="1919")
        if league_id != st.session_state.get("league_id", ""):
            st.session_state.league_id = league_id
        if st.button("Fetch League Rosters"):
            try:
                fetched_path = fetch_league_rosters(league_id, "ottoneu")
                st.session_state.league_roster_fetch_path = fetched_path
                st.session_state.data_refresh_token = st.session_state.get("data_refresh_token", 0) + 1
                cached_values.clear()
                cached_player_trends.clear()
                st.success(f"Fetched league {league_id}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    active_roster_caption = selected_roster_path(st.session_state.get("data_source", default_data_source()))
    if active_roster_caption:
        st.caption(f"Roster file: {active_roster_caption}")


data_source, refresh_token = selected_data_source()
active_roster_path = selected_roster_path(data_source)


tab_stock, tab_trade, tab_available, tab_trends, tab_prospects, tab_auction, tab_simlab, tab_audit, tab_keepcut = st.tabs([
    "Stock Board", "Trade Evaluator", "Available Players", "Player Trends", "Prospects", "Live Auction", "Sim Lab", "Roster Audit", "Keep/Cut Upload"
])

with tab_stock:
    values, _ = cached_values(data_source, refresh_token, active_roster_path)
    mode = st.selectbox("Board", ["underpriced", "mlb risers", "mlb fallers", "sell-high", "buy-low"])
    limit = st.slider("Rows", 10, 100, 25, 5)
    sort_map = {
        "underpriced": ("Current_Surplus", False),
        "mlb risers": ("Stock_Change", False),
        "mlb fallers": ("Stock_Change", True),
        "sell-high": ("Stock_Change", False),
        "buy-low": ("Stock_Change", True),
    }
    sort_by, ascending = sort_map[mode]
    board = values.copy()
    if mode in {"mlb risers", "mlb fallers"}:
        board = board[board["Value_Source"].eq("MLB ROS Auction")]
    board = board.sort_values(sort_by, ascending=ascending).head(limit)
    st.dataframe(
        board[
            [
                "Team Name", "Name", "Positions", "Salary", "Current_Value",
                "Current_Surplus", "Stock_Change", "YTD_Value", "YTD_ROS_Gap",
                "Stock_Label", "Role_Change", "MLB_Status", "Status_Flag",
                "Latest_Transaction_Date", "Value_Source", "Confidence_Label",
            ]
        ],
        use_container_width=True,
    )

with tab_trade:
    values, _ = cached_values(data_source, refresh_token, active_roster_path)
    teams = sorted(values["Team Name"].dropna().astype(str).unique())
    team = st.selectbox("Your team", teams, key="trade_team")
    trade_eval_tab, package_tab = st.tabs([
        "Evaluate", "Mock Packages"
    ])

    with trade_eval_tab:
        from_team = st.selectbox("Incoming player's current team", [""] + teams)
        send = st.text_input("Send", placeholder="Player A, Player B")
        receive = st.text_input("Receive", placeholder="Player C, Player D")
        if st.button("Evaluate Trade"):
            try:
                result = evaluate_trade(
                    values,
                    team_a=team,
                    sends=parse_players(send),
                    receives=parse_players(receive),
                    team_b=from_team or None,
                )
                st.metric("Verdict", result["verdict"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Value Delta", f"{result['value_delta']:+.2f}")
                c2.metric("Salary Delta", f"${result['salary_delta']:+.2f}")
                c3.metric("Surplus Delta", f"{result['surplus_delta']:+.2f}")
                st.subheader("Outgoing")
                st.dataframe(result["outgoing"], use_container_width=True)
                st.subheader("Incoming")
                st.dataframe(result["incoming"], use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    with package_tab:
        package_cols = st.columns([2, 1, 1, 1])
        target_player = package_cols[0].text_input("Target player/package", placeholder="Player A or Player A, Player B")
        target_team = package_cols[1].selectbox("Target team", [""] + teams)
        max_package_size = package_cols[2].selectbox("Max package", [1, 2, 3], index=2)
        package_limit = package_cols[3].slider("Packages", 5, 30, 12, 1)
        include_negative = st.checkbox("Allow negative-surplus outgoing players", value=False)
        if st.button("Generate Mock Packages"):
            try:
                packages = recommend_trade_packages(
                    values,
                    team=team,
                    target_player=target_player,
                    target_team=target_team or None,
                    max_package_size=max_package_size,
                    limit=package_limit,
                    include_negative_surplus=include_negative,
                )
                if packages.empty:
                    st.info("No reasonable packages found for that target.")
                else:
                    show_cols = [
                        "Target", "Target_Team", "Package", "Package_Size", "Package_Tier",
                        "Outgoing_Value", "Receive_Value", "Value_Delta",
                        "Outgoing_Surplus", "Receive_Surplus", "Surplus_Delta",
                        "Salary_Delta", "Opponent_Salary_Delta", "Opponent_Surplus_Delta",
                        "Opponent_Post_Salary", "Opponent_Post_Roster", "Opponent_Context", "Fit_Score",
                    ]
                    st.dataframe(packages[show_cols], use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(str(exc))

with tab_available:
    active_paths = resolve_data_paths(data_source)
    st.subheader("Best Available Players in Your League")
    st.caption("Uses the League Roster source selected in the sidebar.")
    waiver_cols = st.columns(3)
    waiver_limit = waiver_cols[0].slider("Available rows", 10, 100, 40, 5)
    min_value = waiver_cols[1].number_input("Min ROS value", value=-5.0, step=1.0)
    position_filter = waiver_cols[2].multiselect("Positions", ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"], key="available_positions")
    if not active_roster_path:
        st.info("Choose a league roster source in the sidebar to build the available-player board.")
    else:
        try:
            waiver = build_waiver_wire_board(
                rosters=active_roster_path,
                hitters=active_paths.hitters_ros,
                pitchers=active_paths.pitchers_ros,
                avg=active_paths.avg_values,
                mlb_stock=active_paths.mlb_stock,
                mlb_status=active_paths.mlb_status,
                min_value=min_value,
                limit=200,
            )
            if position_filter:
                waiver = waiver[
                    waiver["Positions"].fillna("").apply(
                        lambda value: bool(set(position_filter) & {
                            part.strip().upper() for part in str(value).replace(",", "/").split("/") if part.strip()
                        })
                    )
                ]
            waiver = waiver.head(waiver_limit)
            st.dataframe(
                waiver[
                    [
                        "Player", "Team", "Current_Value", "Available_Surplus",
                        "Composite_Score", "MLB_Stock_Change", "YTD_ROS_Gap",
                        "Roster%", "Stock_Label", "Role_Change", "MLB_Status", "Status_Flag",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(str(exc))

with tab_trends:
    trends = cached_player_trends(data_source, refresh_token, active_roster_path)
    history = cached_trend_history(refresh_token)
    st.caption("Player search shows current ROS/context value and, when snapshots exist, movement from the saved 2025 full-season FGPTS performance baseline to now.")
    trend_cols = st.columns([2, 1, 1, 1])
    query = trend_cols[0].text_input("Player search", placeholder="Ketel Marte")
    teams = sorted(trends["Team Name"].dropna().astype(str).unique())
    team_filter = trend_cols[1].selectbox("Team scope", ["All teams"] + teams)
    trend_mode = trend_cols[2].selectbox("Board", ["risers", "warnings", "elite holds"])
    limit = trend_cols[3].slider("Rows", 10, 75, 25, 5)

    if query:
        try:
            row = resolve_trend_player(
                trends,
                query,
                team=None if team_filter == "All teams" else team_filter,
            )
            st.subheader(f"{row['Name']} - {row['Player_Role']}")
            rank_text = row.get("Eligible_Position_Ranks", "NA")
            if pd.isna(rank_text) or not rank_text:
                rank_text = "NA"
            st.caption(f"{row['Team Name']} | {row['Positions']}")
            status_flag = str(row.get("Status_Flag", "") or "")
            mlb_status = str(row.get("MLB_Status", "") or "")
            if status_flag == "SENT_DOWN":
                transaction_date = row.get("Latest_Transaction_Date", "")
                description = row.get("Latest_Transaction_Description", "")
                st.warning(f"Sent down / minors: {transaction_date} - {description}")
            elif status_flag == "IL" or mlb_status == "INJURED_LIST":
                transaction_date = row.get("Latest_Transaction_Date", "")
                description = row.get("Latest_Transaction_Description", "")
                st.warning(f"Injured list: {transaction_date} - {description}")
            elif mlb_status and mlb_status not in {"Unknown", "ACTIVE_MLB"}:
                st.info(f"MLB status: {mlb_status} / {status_flag}")
            st.write(f"**Eligible ranks:** {rank_text}")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Team", row["Team Name"])
            m2.metric("Position", row["Positions"])
            m3.metric("Role", row["Player_Role"])
            m4.metric("ROS $", f"{row['Current_Value']:.1f}")
            m5.metric("Context $", f"{row['Context_Value']:.1f}", f"{row['Trend_Trade_Adjustment']:+.1f}")
            m6.metric("YTD $", f"{row['YTD_Value']:.1f}" if pd.notna(row.get("YTD_Value")) else "NA")

            prod_cols = st.columns(4)
            if pd.notna(row.get("YTD_PA")):
                slash = "NA"
                if all(pd.notna(row.get(col)) for col in ["YTD_AVG", "YTD_OBP", "YTD_SLG"]):
                    slash = f"{row['YTD_AVG']:.3f}/{row['YTD_OBP']:.3f}/{row['YTD_SLG']:.3f}"
                counting = "NA"
                if any(pd.notna(row.get(col)) for col in ["YTD_HR", "YTD_RBI", "YTD_SB", "YTD_R"]):
                    def count_value(col):
                        value = row.get(col)
                        return float(value) if pd.notna(value) else 0.0

                    counting = (
                        f"{count_value('YTD_HR'):.0f} HR, "
                        f"{count_value('YTD_RBI'):.0f} RBI, "
                        f"{count_value('YTD_SB'):.0f} SB, "
                        f"{count_value('YTD_R'):.0f} R"
                    )
                prod_cols[0].metric("YTD PA", f"{row['YTD_PA']:.0f}")
                prod_cols[1].metric("AVG/OBP/SLG", slash)
                prod_cols[2].metric("HR/RBI/SB/R", counting)
                prod_cols[3].metric("SC xwOBA", f"{row['SC_xwOBA']:.3f}" if pd.notna(row.get("SC_xwOBA")) else "NA")
            else:
                prod_cols[0].metric("YTD G/IP", f"{row['YTD_G']:.0f}/{row['YTD_IP']:.1f}" if pd.notna(row.get("YTD_G")) and pd.notna(row.get("YTD_IP")) else "NA")
                prod_cols[1].metric("YTD GS", f"{row['YTD_GS']:.0f}" if pd.notna(row.get("YTD_GS")) else "NA")
                prod_cols[2].metric("SC xwOBAA", f"{row['SC_xwOBA_Allowed']:.3f}" if pd.notna(row.get("SC_xwOBA_Allowed")) else "NA")
                prod_cols[3].metric("Skill", f"{row['Skill_Score']:.2f}" if pd.notna(row.get("Skill_Score")) else "NA")

            st.write(f"**Trend:** {row['Trend_Label']} | **Sample:** {row['Trend_Sample']} | **Notes:** {row['Trend_Notes']}")
            detail_cols = [
                "Team Name", "Name", "Positions", "Eligible_Position_Ranks", "Primary_Position", "Position_Rank", "Player_Role",
                "Salary", "Current_Value", "Context_Value",
                "Trend_Trade_Adjustment", "Trend_Label", "Trend_Sample", "YTD_Value",
                "YTD_ROS_Gap", "Projection_Change", "Skill_Score", "YTD_AVG", "YTD_OBP",
                "YTD_SLG", "YTD_HR", "YTD_RBI", "YTD_SB", "YTD_R", "SC_xwOBA",
                "SC_xwOBA_Allowed", "Role_Change",
                "MLB_Status", "Status_Flag", "Latest_Transaction_Date", "Latest_Transaction_Description",
                "Stock_Label", "Prospect_Pedigree_Label", "PS_Best_Score", "PS_Best_Year",
                "Confidence_Label", "Value_Source",
            ]
            detail = pd.DataFrame([row])[[col for col in detail_cols if col in trends.columns]]
            st.dataframe(detail, use_container_width=True, hide_index=True)
            waterfall = trend_waterfall_rows(row).set_index("Component")
            st.bar_chart(waterfall)
            if not history.empty:
                ph = player_history(
                    history,
                    query,
                    team=None if team_filter == "All teams" else team_filter,
                )
                if len(ph) >= 2:
                    st.subheader("Trend Graph")
                    st.caption("Orange line is context value. Gray range shows ROS-to-context spread at each snapshot; blue tick is ROS value.")
                    chart = trend_value_chart(ph)
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.caption("No numeric value history is available for this player yet.")
                elif len(ph) == 1:
                    st.caption("Only one saved snapshot exists for this player so far.")
            else:
                st.caption("No saved trend snapshots yet. Build snapshots to unlock player history.")
        except Exception as exc:
            st.error(str(exc))

    if trend_mode == "risers":
        board = trends.sort_values("Trend_Trade_Adjustment", ascending=False)
    elif trend_mode == "warnings":
        board = trends.sort_values("Trend_Trade_Adjustment", ascending=True)
    else:
        board = trends[
            (trends["Current_Value"] >= 20)
            & (trends["Trend_Trade_Adjustment"] < 0)
        ].sort_values("Current_Value", ascending=False)

    board_cols = [
        "Team Name", "Name", "Positions", "Eligible_Position_Ranks", "Primary_Position", "Position_Rank", "Player_Role",
        "Salary", "Current_Value", "Context_Value",
        "Trend_Trade_Adjustment", "Trend_Label", "Trend_Sample", "YTD_Value",
        "YTD_ROS_Gap", "Skill_Score", "Role_Change", "MLB_Status", "Status_Flag",
        "Latest_Transaction_Date", "Prospect_Pedigree_Label", "Trend_Notes",
    ]
    st.dataframe(board.head(limit)[[col for col in board_cols if col in board.columns]], use_container_width=True, hide_index=True)

    if not history.empty:
        movers = latest_movement(history)
        if not movers.empty:
            st.subheader("Latest Snapshot Movers")
            mover_cols = [
                "Snapshot", "Team Name", "Name", "Positions", "Eligible_Position_Ranks", "Primary_Position",
                "Position_Rank", "Player_Role", "Current_Value",
                "Context_Value", "Context_Value_Delta", "Skill_Score_Delta",
                "Trend_Label", "Prospect_Pedigree_Label", "Trend_Notes",
            ]
            st.dataframe(movers.head(limit)[[col for col in mover_cols if col in movers.columns]], use_container_width=True, hide_index=True)

with tab_prospects:
    updates = cached_prospect_updates(data_source, refresh_token)
    control_cols = st.columns(3)
    mode = control_cols[0].selectbox("Prospect view", ["risers", "fallers", "top values"])
    position_options = sorted({
        position.strip()
        for value in updates["Pos"].dropna().astype(str)
        for position in value.replace("/", ",").split(",")
        if position.strip()
    })
    positions = control_cols[1].multiselect("Positions", position_options)
    limit = control_cols[2].slider("Prospect rows", 10, 100, 25, 5)
    if positions:
        pattern = "|".join([rf"\b{position}\b" for position in positions])
        updates = updates[updates["Pos"].fillna("").astype(str).str.contains(pattern, case=False, regex=True)]
    if mode == "risers":
        updates = updates.sort_values("Prospect_Update", ascending=False)
    elif mode == "fallers":
        updates = updates.sort_values("Prospect_Update", ascending=True)
    else:
        updates = updates.sort_values("Updated_Prospect_Value", ascending=False)
    prospect_columns = [
        "Name", "Org", "Savant_Level", "Player_Type", "Pos", "Age", "Pitches",
        "K_Rate", "BB_Rate", "Whiff_Rate", "SwStr_Rate",
        "Pitching_Plus", "Stuff_Plus", "Command_Plus", "Stuff_Pitches",
        "PS Score", "xwOBA", "Evidence_Score", "Confidence_Label",
        "Pipeline_Rank", "Composite_Rank", "Prior_Value",
        "Prospect_Update", "Updated_Prospect_Value", "Match_Status",
    ]
    visible_columns = [col for col in prospect_columns if col in updates.columns]
    st.dataframe(updates.head(limit)[visible_columns], use_container_width=True)

with tab_auction:
    room = auction_room()
    st.caption("Streamlit auction mode uses the same AI valuation, roster, budget, recommendation, history, and simulation logic as the terminal auction.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pick", len(room.history_rows))
    c2.metric("Available", len(room.pool))
    c3.metric("Your Budget", room.agents[room.human_team].budget)
    c4.metric("Your Roster", f"{roster_size(room.agents[room.human_team])}/{TOTAL_ROSTER_LIMIT}")

    controls = st.columns(5)
    if controls[0].button("New Auction"):
        st.session_state.auction_room = LiveAuctionRoom(
            pool=cached_player_pool(data_source, refresh_token),
            human_team=0,
            bid_seconds=60,
            auto_human=False,
        )
        st.session_state.auction_current = None
        st.rerun()
    sim_count = controls[1].number_input("Sim picks", min_value=1, max_value=480, value=10)
    if controls[2].button("Sim X"):
        if st.session_state.get("auction_current") is not None:
            st.session_state.auction_status = "Resolve the current nomination before simulating future picks."
        else:
            before = len(room.history_rows)
            room.simulate_picks(int(sim_count))
            st.session_state.auction_status = f"Simulated {len(room.history_rows) - before} picks."
        st.rerun()
    if controls[3].button("Sim End"):
        if st.session_state.get("auction_current") is not None:
            st.session_state.auction_status = "Resolve the current nomination before simulating to the end."
        else:
            before = len(room.history_rows)
            room.simulate_to_end()
            st.session_state.auction_status = f"Simulated {len(room.history_rows) - before} picks. Auction now has {len(room.history_rows)} completed picks."
        st.rerun()
    if controls[4].button("Save Reports"):
        room.save_reports()
        st.success("Saved live_auction_results.csv and live_team_rosters.csv")
    if st.session_state.get("auction_status"):
        st.info(st.session_state.auction_status)

    st.subheader("Nomination / Bidding")
    current = st.session_state.get("auction_current")
    if current is None:
        nominator = room.nomination_cursor
        if nominator == room.human_team:
            search = st.text_input("Nomination search", placeholder="Type a player name, e.g. Ohtani")
            legal_players = [player for player in room.pool if room.team_can_roster(room.human_team, player)]
            if search:
                search_text = search.strip().lower()
                legal_players = [player for player in legal_players if search_text in player["Name"].lower()]
            legal_players = sorted(legal_players, key=lambda player: float(player["dollars"]), reverse=True)
            options = [player["Name"] for player in legal_players[:250]]
            recs = room.recommendation_rows(limit=20)
            with st.expander("Recommended nominations", expanded=False):
                st.dataframe(pd.DataFrame(recs), use_container_width=True)
            if options:
                choice = st.selectbox("Your nomination", options)
                if st.button("Nominate Selected"):
                    start_next_nomination(room, human_choice=choice)
                    run_ai_round(room)
                    complete_if_no_live_bidders(room)
                    st.rerun()
            else:
                st.info("No legal nomination targets match that search.")
        else:
            st.write(f"Next nominator: {room.team_label(nominator)}")
            if st.button("Start Next Nomination"):
                start_next_nomination(room)
                run_ai_round(room)
                complete_if_no_live_bidders(room)
                st.rerun()
    else:
        player = current["player"]
        leader = room.team_label(current["high_bidder"]) if current["high_bidder"] is not None else "No bids"
        st.write(f"**{room.team_label(current['nominator'])} nominated {player['Name']}**")
        st.write(f"Positions: {'/'.join(player['positions'])} | Model value: ${player['dollars']:.2f}")
        st.metric("High Bid", f"${current['current_bid']} by {leader}")
        st.dataframe(pd.DataFrame(current.get("bid_log", [])), use_container_width=True, hide_index=True)
        next_bid = current["current_bid"] + 1
        human_max = max_legal_bid_for_player(room.agents[room.human_team], player)
        bid_cols = st.columns(4)
        if bid_cols[0].button(f"Bid ${next_bid}", disabled=next_bid > human_max or current["high_bidder"] == room.human_team):
            current["current_bid"] = next_bid
            current["high_bidder"] = room.human_team
            current["human_passed"] = False
            current.setdefault("bid_log", []).append({
                "Bid": current["current_bid"],
                "Team": room.team_label(room.human_team),
                "Action": "human bid",
            })
            run_ai_round(room)
            complete_if_no_live_bidders(room)
            st.rerun()
        custom_bid = bid_cols[1].number_input("Custom bid", min_value=1, max_value=max(human_max, 1), value=min(max(next_bid, 1), max(human_max, 1)))
        if bid_cols[2].button("Submit Bid", disabled=custom_bid <= current["current_bid"] or custom_bid > human_max):
            current["current_bid"] = int(custom_bid)
            current["high_bidder"] = room.human_team
            current["human_passed"] = False
            current.setdefault("bid_log", []).append({
                "Bid": current["current_bid"],
                "Team": room.team_label(room.human_team),
                "Action": "human custom bid",
            })
            run_ai_round(room)
            complete_if_no_live_bidders(room)
            st.rerun()
        if bid_cols[3].button("Pass"):
            current["human_passed"] = True
            current["passed"].add(room.human_team)
            current.setdefault("bid_log", []).append({
                "Bid": current["current_bid"],
                "Team": room.team_label(room.human_team),
                "Action": "human pass",
            })
            run_ai_round(room)
            if room.no_live_bidders(player, current["ai_limits"], current["current_bid"], current["high_bidder"], current["passed"], current["human_passed"]):
                complete_streamlit_auction(room)
            st.rerun()
        if st.button("Sell Now / Resolve"):
            complete_streamlit_auction(room)
            st.rerun()

    st.subheader("Auction Commands")
    view_cols = st.columns(5)
    view = view_cols[0].selectbox("View", ["recommend", "rosters", "team roster", "my roster", "history", "sold by position"])
    position = view_cols[1].selectbox("Position", ["All", "SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "MI", "CI", "UTIL"])
    selected_team = view_cols[2].selectbox("Team", [room.team_label(idx) for idx in range(12)])
    limit = view_cols[3].slider("Limit", 5, 100, 10, 5)
    mode = view_cols[4].selectbox("Mode", ["AUTO", "EARLY", "MID", "LATE", "PROSPECTS", "ALL"])
    selected_position = None if position == "All" else position
    if view == "recommend":
        st.dataframe(pd.DataFrame(room.recommendation_rows(position=selected_position, limit=limit, mode=mode)), use_container_width=True)
    elif view == "rosters":
        st.dataframe(auction_summary(room), use_container_width=True)
    elif view == "team roster":
        team_idx = int(selected_team.split()[-1]) - 1
        agent = room.agents[team_idx]
        roster_scope = st.radio(
            "Roster scope",
            ["Active lineup", "Full roster", "Bench/reserves"],
            horizontal=True,
        )
        st.caption(
            f"{selected_team}: {roster_size(agent)}/{TOTAL_ROSTER_LIMIT} players | "
            f"Budget ${agent.budget} | Max bid ${max_legal_bid(agent)}"
        )
        if roster_scope == "Active lineup":
            st.dataframe(team_active_lineup_dataframe(room, team_idx), use_container_width=True, hide_index=True)
        elif roster_scope == "Bench/reserves":
            roster = team_roster_dataframe(room, team_idx)
            st.dataframe(roster[roster["Slot"].eq("Bench")], use_container_width=True, hide_index=True)
        else:
            st.dataframe(team_roster_dataframe(room, team_idx), use_container_width=True, hide_index=True)
    elif view == "my roster":
        st.dataframe(team_active_lineup_dataframe(room, room.human_team), use_container_width=True, hide_index=True)
    elif view == "history":
        st.dataframe(pd.DataFrame(room.history_rows[-limit:]), use_container_width=True)
    elif view == "sold by position":
        rows = [
            row for row in room.roster_rows
            if selected_position is None
            or row["Slot"].upper() == selected_position
            or selected_position in {p.upper() for p in row["Positions"].split("/")}
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_simlab:
    st.caption("Run independent full-auction simulations and summarize the roster-construction styles AI teams settle into.")
    lab_cols = st.columns(5)
    sim_count = lab_cols[0].slider("Auction sims", 1, 50, 3, 1)
    max_picks = lab_cols[1].selectbox("Depth", ["Smoke 60 picks", "Quick 120 picks", "Half 240 picks", "Full 480 picks"])
    max_picks_value = {
        "Smoke 60 picks": 60,
        "Quick 120 picks": 120,
        "Half 240 picks": 240,
        "Full 480 picks": None,
    }[max_picks]
    seed = lab_cols[2].number_input("Seed", min_value=0, max_value=1_000_000, value=42)
    use_bandit = lab_cols[3].checkbox("Train bid bandit")
    run_lab = lab_cols[4].button("Run Sim Lab")
    if run_lab:
        progress = st.progress(0)
        status = st.empty()

        def update_progress(done, total):
            progress.progress(done / total)
            status.info(f"Completed {done}/{total} auction sims...")

        if use_bandit:
            team_runs, strategy_summary, action_summary, decisions = run_bandit_training(
                sim_count=int(sim_count),
                batters=resolve_data_paths(data_source).batters_auction,
                pitchers=resolve_data_paths(data_source).pitchers_auction,
                prospects=resolve_data_paths(data_source).prospects,
                seed=int(seed),
                max_picks=max_picks_value,
                progress_callback=update_progress,
            )
            st.session_state.simlab_action_summary = action_summary
            st.session_state.simlab_decisions = decisions
        else:
            team_runs, strategy_summary, metric_summary = run_auction_sims(
                sim_count=int(sim_count),
                batters=resolve_data_paths(data_source).batters_auction,
                pitchers=resolve_data_paths(data_source).pitchers_auction,
                prospects=resolve_data_paths(data_source).prospects,
                seed=int(seed),
                max_picks=max_picks_value,
                progress_callback=update_progress,
            )
            st.session_state.simlab_action_summary = None
            st.session_state.simlab_decisions = None
        status.success(f"Completed {sim_count} {max_picks.lower()} simulations.")
        st.session_state.simlab_team_runs = team_runs
        st.session_state.simlab_strategy_summary = strategy_summary
        if not use_bandit:
            st.session_state.simlab_metric_summary = metric_summary
        else:
            metrics = [
                "Stars_40", "Cheap_5", "Mid_6_25", "Prospects", "Prospect_Salary",
                "Active_Salary", "Bench_Salary", "Total_Value", "Total_Surplus",
                "Alpha", "Beta",
            ]
            st.session_state.simlab_metric_summary = (
                team_runs.groupby("Strategy")[metrics].mean().round(2).reset_index()
            )

    if "simlab_team_runs" in st.session_state:
        team_runs = st.session_state.simlab_team_runs
        strategy_summary = st.session_state.simlab_strategy_summary
        metric_summary = st.session_state.simlab_metric_summary

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Team Sims", len(team_runs))
        c2.metric("Avg Stars $40+", f"{team_runs['Stars_40'].mean():.2f}")
        c3.metric("Avg Cheap <= $5", f"{team_runs['Cheap_5'].mean():.2f}")
        c4.metric("Avg Prospects", f"{team_runs['Prospects'].mean():.2f}")

        st.subheader("Strategy Frequency")
        st.dataframe(strategy_summary, use_container_width=True, hide_index=True)

        st.subheader("Average Metrics by Strategy")
        st.dataframe(metric_summary, use_container_width=True, hide_index=True)

        if st.session_state.get("simlab_action_summary") is not None:
            st.subheader("Bandit Action Feedback")
            st.dataframe(st.session_state.simlab_action_summary, use_container_width=True, hide_index=True)

            st.subheader("Bandit Purchase Decisions")
            st.dataframe(
                st.session_state.simlab_decisions.tail(250),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Team-Sim Detail")
        strategy_filter = st.multiselect("Strategies", sorted(team_runs["Strategy"].unique()))
        detail = team_runs
        if strategy_filter:
            detail = detail[detail["Strategy"].isin(strategy_filter)]
        st.dataframe(
            detail.sort_values(["Sim", "Team"])[
                [
                    "Sim", "Team", "Strategy", "Stars_40", "Cheap_5", "Mid_6_25",
                    "Prospects", "Prospect_Salary", "Active_Salary", "Bench_Salary",
                    "Total_Value", "Total_Salary", "Total_Surplus", "Alpha", "Beta",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab_audit:
    values, _ = cached_values(data_source, refresh_token, active_roster_path)
    teams = sorted(values["Team Name"].dropna().astype(str).unique())
    audit_team = st.selectbox("Audit team", teams)
    audit = build_roster_audit(values, audit_team)

    overview = audit["overview"].iloc[0]
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Roster", int(overview["Roster_Size"]))
    a2.metric("Salary", f"${overview['Salary']:.0f}")
    a3.metric("Value", f"{overview['Current_Value']:.1f}")
    a4.metric("Surplus", f"{overview['Current_Surplus']:+.1f}")
    a5.metric("Needs", int(overview["Needs"]))

    st.subheader("Position Audit")
    st.dataframe(
        audit["positions"][
            [
                "Position", "Filled", "Slots", "Team_Value", "Expected_Position_Value",
                "Value_Gap", "Team_Surplus", "Need_Level",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Category Audit")
    st.dataframe(audit["categories"], use_container_width=True, hide_index=True)

    st.subheader("Trade Chips")
    st.dataframe(
        audit["trade_chips"].head(25),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Trade Ideas")
    st.dataframe(
        audit["trade_ideas"],
        use_container_width=True,
        hide_index=True,
    )

with tab_keepcut:
    st.caption("Load an Ottoneu roster export, then select a team for keep/cut and arbitration analysis.")
    active_paths = resolve_data_paths(data_source)
    keepcut_source = st.radio(
        "Roster source",
        ["Upload CSV", "Ottoneu league number"],
        horizontal=True,
        key="keepcut_roster_source",
    )
    roster_path = None
    if keepcut_source == "Upload CSV":
        roster_upload = st.file_uploader("Ottoneu roster CSV", type="csv", key="keepcut_roster_upload")
        roster_path = write_upload(roster_upload, "uploaded_rosters.csv")
    else:
        league_id = st.text_input(
            "League number",
            value=st.session_state.get("keepcut_league_id", ""),
            placeholder="1919",
            key="keepcut_league_id_input",
        )
        if league_id != st.session_state.get("keepcut_league_id", ""):
            st.session_state.keepcut_league_id = league_id
        fetch_cols = st.columns([1, 3])
        if fetch_cols[0].button("Fetch rosters", key="keepcut_fetch_rosters"):
            try:
                fetched_path = fetch_league_rosters(league_id, "keepcut_ottoneu")
                st.session_state.keepcut_roster_fetch_path = fetched_path
                st.success(f"Fetched league {league_id}.")
            except Exception as exc:
                st.error(str(exc))
        roster_path = st.session_state.get("keepcut_roster_fetch_path")
        if roster_path:
            fetch_cols[1].caption(f"Roster file: {roster_path}")

    hitters_upload = st.file_uploader("Optional hitters ROS CSV", type="csv")
    pitchers_upload = st.file_uploader("Optional pitchers ROS CSV", type="csv")
    avg_upload = st.file_uploader("Optional FGPTS average values CSV", type="csv")
    if roster_path:
        hitters_path = write_upload(hitters_upload, "uploaded_hitters_ros.csv") or active_paths.hitters_ros
        pitchers_path = write_upload(pitchers_upload, "uploaded_pitchers_ros.csv") or active_paths.pitchers_ros
        avg_path = write_upload(avg_upload, "uploaded_avgvalues.csv") or active_paths.avg_values
        try:
            report, summary = build_reports(
                roster_path,
                hitters_path,
                pitchers_path,
                avg_path,
                active_paths.prospects,
                active_paths.mlb_stock,
            )
            teams = sorted(summary["Team Name"].dropna().astype(str).unique())
            team = st.selectbox("Team", teams, key="keepcut_team")
            resolved_team = resolve_team_name(report, team)
            team_report = report[report["Team Name"].eq(resolved_team)]
            st.subheader("Team Summary")
            st.dataframe(summary[summary["Team Name"].eq(resolved_team)], use_container_width=True)
            st.subheader("Keep/Cut/Hold")
            st.dataframe(
                team_report[
                    [
                        "Name", "Positions", "Salary", "Future_Value", "Future_Surplus",
                        "YTD_Value", "YTD_ROS_Gap", "Stock_Label", "Role_Change",
                        "Active_Slot", "MLB_Level", "Is_Prospect", "Recommendation",
                    ]
                ].sort_values(["Recommendation", "Future_Surplus"], ascending=[True, False]),
                use_container_width=True,
            )
            st.subheader("Arbitration Targets")
            st.dataframe(build_arbitration_report(report, target_team=resolved_team, limit=50), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
    else:
        if keepcut_source == "Upload CSV":
            st.info("Upload an Ottoneu roster CSV to build a keep/cut report.")
        else:
            st.info("Enter an Ottoneu league number and fetch rosters to select a team.")
