"""Exploratory time, competition, confidence and selected-error profiles."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.probability_metrics import binary_log_loss_vector

EPS = 1e-12


def _metrics(frame: pd.DataFrame, p: str = "p_event", y: str = "y_event") -> dict:
    d = frame[[p, y]].dropna()
    if d.empty:
        return {"n": 0, "expected_wins": None, "actual_wins": None, "signed_error_sum": None,
                "signed_error_per_fixture": None, "log_loss": None, "brier": None}
    pred = d[p].to_numpy(float).clip(EPS, 1-EPS)
    target = d[y].to_numpy(float)
    ll = binary_log_loss_vector(target, pred, epsilon=EPS)
    brier = (target-pred)**2
    return {"n": len(d), "expected_wins": float(pred.sum()), "actual_wins": float(target.sum()),
            "signed_error_sum": float((pred-target).sum()),
            "signed_error_per_fixture": float((pred-target).mean()),
            "log_loss": float(ll.mean()), "brier": float(brier.mean())}
def _profile(frame, column, label):
    rows=[]
    for key, group in frame.groupby(column, dropna=False, observed=False, sort=True):
        row={"profile":label,"group":str(key)} | _metrics(group)
        if "q_consensus_a" in group:
            market_ll=_metrics(group,"q_consensus_a")["log_loss"]
            market_brier=_metrics(group,"q_consensus_a")["brier"]
            row.update({"market_log_loss":market_ll,"market_brier":market_brier,
                        "model_minus_market_ll":row["log_loss"]-market_ll,
                        "excess_log_loss_sum":len(group)*(row["log_loss"]-market_ll),
                        "excess_brier_sum":len(group)*(row["brier"]-market_brier)})
        rows.append(row)
    return rows


def _month_interval(df, value_col, months):
    monthly = df.groupby("month")[value_col].agg(["sum", "size"]).reindex(months, fill_value=0)
    return block_ratio_interval(monthly["sum"], monthly["size"], monthly.index, draws=5000, seed=20260930)


def _pb_tail(frame, threshold):
    d=frame.loc[frame.p_event.ge(threshold)].copy()
    if d.empty: return {"threshold":threshold,"n":0,"observed_losses":0,"expected_losses":0,"poisson_binomial_tail_probability":None}
    p=d.p_event.to_numpy(float)
    observed=int((d.y_event.eq(0)).sum())
    # Heterogeneous-P Poisson-binomial distribution for independent fixtures.
    dist=np.array([1.0])
    for prob in (1-p): dist=np.convolve(dist, np.array([1-prob,prob]))
    tail=float(dist[observed:].sum()) if observed < len(dist) else 0.0
    monthly=d.groupby("month",dropna=False).agg(observed_losses=("y_event",lambda x:int((x==0).sum())),expected_losses=("p_event",lambda x:float((1-x).sum())),n=("y_event","size"))
    months=sorted(frame.month.dropna().astype(str).unique())
    monthly=monthly.reindex(months,fill_value=0)
    interval = block_ratio_interval(
        monthly.observed_losses - monthly.expected_losses,
        monthly.n, monthly.index, draws=5000, seed=20260930,
    )
    nblock = len(months)
    ci = [interval["ci_lower"], interval["ci_upper"]] if interval["status"] == "ESTIMATED" else None
    return {"threshold":threshold,"n":len(d),"observed_losses":observed,"expected_losses":float((1-p).sum()),"observed_minus_expected":float(observed-(1-p).sum()),"observed_loss_rate":observed/len(d),"expected_loss_rate":float((1-p).mean()),"poisson_binomial_tail_probability_independence":tail,"independence_caveat":"Exact heterogeneous-p Poisson-binomial assumes independent fixtures; shared teams/tournaments/time violate this. Monthly-block interval describes observed-minus-expected loss-rate stability, not an independence-corrected tail p-value.","monthly_block_residual_rate_ci95":ci,"monthly_blocks_universe":nblock}


def run(tables: dict[str,pd.DataFrame], protocol:dict, output:Path)->dict:
    output.mkdir(parents=True,exist_ok=True)
    opp=tables["opportunities"]
    primary=opp.loc[opp.cohort.eq("oddspapi_common") & opp.source.eq("best_price") & opp.horizon.eq("T-1h") & opp.audit_primary_eligible].copy()
    if primary.fixture_id.duplicated().any(): raise ValueError("Primary opportunity frame is not one row per fixture")
    primary["p_canonical"]=primary.p_event.astype(float)
    primary["y_canonical"]=primary.y_event.astype(int)
    primary["q_canonical"]=primary.q_consensus_a.astype(float)
    primary["favorite_is_a"]=primary.p_canonical.ge(.5)
    primary["p_event"]=np.maximum(primary.p_canonical,1-primary.p_canonical)
    primary["y_event"]=np.where(primary.favorite_is_a,primary.y_canonical,1-primary.y_canonical)
    primary["q_consensus_a"]=np.where(primary.favorite_is_a,primary.q_canonical,1-primary.q_canonical)
    primary["canonical_signed_error"]=primary.p_canonical-primary.y_canonical
    primary_ll=binary_log_loss_vector(primary.y_event, primary.p_event, epsilon=EPS)
    market_ll=binary_log_loss_vector(primary.y_event, primary.q_consensus_a, epsilon=EPS)
    primary["excess_log_loss_contribution"]=primary_ll-market_ll
    primary["excess_brier_contribution"]=(primary.p_event-primary.y_event)**2-(primary.q_consensus_a-primary.y_event)**2
    primary["month"]=pd.to_datetime(primary.date,utc=True).dt.strftime("%Y-%m")
    primary["confidence"]=np.maximum(primary.p_event,1-primary.p_event)
    primary["market_probability"]=np.maximum(primary.q_consensus_a,1-primary.q_consensus_a)
    primary["disagreement"]=abs(primary.p_event-primary.q_consensus_a)
    primary["confidence_band"]=pd.cut(primary.confidence,[.5,.6,.7,.8,.9,1.0000001],right=False,labels=[".5-.6",".6-.7",".7-.8",".8-.9",".9-1"])
    primary["disagreement_band"]=pd.cut(primary.disagreement,[-1e-12,.05,.10,.20,1.000001],right=False,labels=["0-.05",".05-.10",".10-.20",">=.20"])
    primary["market_band"]=pd.cut(primary.market_probability,[.5,.6,.7,.8,.9,1.0000001],right=False,labels=[".5-.6",".6-.7",".7-.8",".8-.9",".9-1"])
    primary["bo_group"]=primary.best_of.map(lambda x: "BO"+str(int(x)) if pd.notna(x) else "unknown")
    bets=tables["bets"]
    selected=bets.loc[bets.cohort.eq("oddspapi_common") & bets.source.eq("best_price") & bets.horizon.eq("T-1h") & bets.quality.eq("exclude_q25_conflicts") & bets.policy.eq("raw_ev_gt_5")].copy()
    if len(selected)!=498: raise ValueError(f"Primary raw selection count changed: {len(selected)}")
    current=bets.loc[bets.cohort.eq("oddspapi_common") & bets.source.eq("best_price") & bets.horizon.eq("T-1h") & bets.quality.eq("exclude_q25_conflicts") & bets.policy.eq("current_application_filters")]
    selected["confidence"]=np.maximum(selected.p_model_bet,1-selected.p_model_bet)
    selected["market_probability"]=np.maximum(selected.q_bet,1-selected.q_bet)
    selected["disagreement"]=abs(selected.p_model_bet-selected.q_bet)
    selected["month"]=pd.to_datetime(selected.date,utc=True).dt.strftime("%Y-%m")
    selected["odds_band"]=pd.cut(selected.odds,[0,1.8,2.5,3.5,5,np.inf],right=False,labels=["<1.8","1.8-2.5","2.5-3.5","3.5-5",">=5"])
    selected["ev_band"]=pd.cut(selected.ev_bet,[.05,.10,np.inf],right=False,labels=["5-10%",">=10%"])
    selected["confidence_band"]=pd.cut(selected.confidence,[.5,.6,.7,.8,.9,1.000001],right=False,labels=[".5-.6",".6-.7",".7-.8",".8-.9",".9-1"])
    selected["disagreement_band"]=pd.cut(selected.disagreement,[-1e-12,.05,.10,.20,1.000001],right=False,labels=["0-.05",".05-.10",".10-.20",">=.20"])
    selected["market_band"]=pd.cut(selected.market_probability,[.5,.6,.7,.8,.9,1.000001],right=False,labels=[".5-.6",".6-.7",".7-.8",".8-.9",".9-1"])
    selected["bo_group"]=selected.best_of.map(lambda x:"BO"+str(int(x)) if pd.notna(x) else "unknown")
    selected["selected_error"]=selected.won.astype(float)-selected.p_for_ev.astype(float)
    selected["realized_net_after_tax"]=.88*pd.to_numeric(selected.odds).astype(float)*selected.won.astype(float)-1
    selected["selected_ev_error"]=selected.realized_net_after_tax-pd.to_numeric(selected.ev_bet).astype(float)
    selected["market_selected_error"]=selected.won.astype(float)-selected.q_bet.astype(float)
    selected["model_market_ll_delta"] = binary_log_loss_vector(selected.won, selected.p_model_bet, epsilon=EPS) - binary_log_loss_vector(selected.won, selected.q_bet, epsilon=EPS)
    primary=primary.merge(selected[["fixture_id","side","odds","ev_bet","won","p_model_bet","q_bet","p_for_ev"]].rename(columns={"side":"selected_side","odds":"selected_odds","ev_bet":"selected_ev","won":"selected_won","p_model_bet":"selected_p","q_bet":"selected_q","p_for_ev":"selected_p_for_ev"}),on="fixture_id",how="left",validate="one_to_one")
    primary["selection_status"]=np.where(primary.selected_side.notna(),"selected","unselected")
    # Canonical target-roster is an explicitly separate retrospective scenario.
    origins=tables["origins"]
    native_diag=origins.loc[origins.cohort.eq("oddspapi_common") & origins.horizon.eq("T-1h"),["fixture_id","team1_id","team2_id","hindsight_any_roster_change","hindsight_same_players","canonical_same_cutoff_day"]].drop_duplicates().rename(columns={"team1_id":"origin_team1_id","team2_id":"origin_team2_id","hindsight_any_roster_change":"origin_hindsight_any_roster_change","hindsight_same_players":"origin_hindsight_same_players","canonical_same_cutoff_day":"origin_canonical_same_cutoff_day"})
    primary=primary.merge(native_diag,on="fixture_id",how="left",validate="one_to_one")
    if primary.origin_team1_id.isna().any(): raise ValueError("Canonical team metadata missing in primary origins")
    primary["canonical_prior_comparable"]=primary.origin_hindsight_same_players.astype(bool)&primary.origin_canonical_same_cutoff_day.astype(bool)
    canon=origins.loc[origins.cohort.eq("oddspapi_common") & origins.horizon.eq("T-1h")].copy()
    canon["canonical_scenario_eligible"]=canon.hindsight_same_players.astype(bool)&canon.canonical_same_cutoff_day.astype(bool)
    # Detailed disjoint group tables; totals include error contribution sums.
    profiles=[]
    for col,label in [("month","month"),("competition_family","competition_family"),("tournament_name","tournament"),("bo_group","best_of"),("confidence_band","confidence"),("disagreement_band","disagreement"),("market_band","market_probability"),("selection_status","selection_status")]: profiles+=_profile(primary,col,label)
    for col,label in [("month","selected_month"),("competition_family","selected_competition_family"),("tournament_name","selected_tournament"),("bo_group","selected_best_of"),("confidence_band","selected_confidence"),("disagreement_band","selected_disagreement"),("market_band","selected_market_probability"),("odds_band","selected_odds"),("ev_band","selected_ev")]:
        for key,g in selected.groupby(col,dropna=False,observed=False,sort=True):
            row={"profile":label,"group":str(key),"n":len(g),"expected_wins":float(g.p_model_bet.sum()),"actual_wins":int(g.won.sum()),"signed_error_sum":float(g.p_model_bet.sum()-g.won.sum()),"signed_error_per_fixture":float((g.p_model_bet-g.won).mean()) if len(g) else None,"selected_ev_error_sum":float(g.selected_ev_error.sum()),"selected_ev_error_per_bet":float(g.selected_ev_error.mean()) if len(g) else None,"log_loss":_metrics(pd.DataFrame({"p_event":g.p_model_bet,"y_event":g.won}))["log_loss"],"brier":_metrics(pd.DataFrame({"p_event":g.p_model_bet,"y_event":g.won}))["brier"],"market_log_loss":_metrics(pd.DataFrame({"p_event":g.q_bet,"y_event":g.won}))["log_loss"],"model_minus_market_ll":None}
            row["model_minus_market_ll"]=row["log_loss"]-row["market_log_loss"] if row["log_loss"] is not None else None
            row["excess_log_loss_sum"]=len(g)*row["model_minus_market_ll"]
            row["market_brier"]=_metrics(pd.DataFrame({"p_event":g.q_bet,"y_event":g.won}))["brier"]
            row["excess_brier_sum"]=len(g)*(row["brier"]-row["market_brier"])
            profiles.append(row)
    pd.DataFrame(profiles).to_csv(output/"profiles.csv",index=False)
    # Internal profiles use the A0 favorite; exported event fields remain canonical A.
    exported_orientation = {
        "p_event": "p_a0_favorite", "y_event": "a0_favorite_won",
        "q_consensus_a": "q_for_a0_favorite",
        "p_canonical": "p_event", "y_canonical": "y_event",
        "q_canonical": "q_consensus_a",
    }
    primary.rename(columns=exported_orientation).to_parquet(output/"primary_fixture_profiles.parquet",index=False)
    selected.to_parquet(output/"selected_fixture_profiles.parquet",index=False)
    # Fixture contribution concentration, along with canonical metadata team IDs/names.
    primary["favorite_probability_error_contribution"]=primary.p_event-primary.y_event
    contrib=primary.sort_values("excess_log_loss_contribution",ascending=False).copy()
    contrib.rename(columns=exported_orientation).to_csv(output/"fixture_error_contributions.csv",index=False)
    # Loss cases paired descriptively to nearest winning controls in same competition/BO/market band.
    selected["miss_severity"]=np.where(selected.won.eq(False),selected.p_model_bet,0.0)
    losers=selected.loc[~selected.won].sort_values("miss_severity",ascending=False).head(40)
    wins=selected.loc[selected.won].copy()
    controls=[]
    for r in losers.itertuples():
        cand=wins.loc[wins.competition_family.eq(r.competition_family)&wins.bo_group.eq(r.bo_group)&wins.market_band.eq(r.market_band)]
        if cand.empty: cand=wins.loc[wins.market_band.eq(r.market_band)]
        if cand.empty: cand=wins
        if not cand.empty:
            c=cand.assign(_dist=(cand.p_model_bet-r.p_model_bet).abs()).sort_values("_dist").iloc[0]
            controls.append({"miss_fixture_id":r.fixture_id,"miss_tournament":r.tournament_name,"miss_competition_family":r.competition_family,"miss_best_of":r.bo_group,"miss_market_band":str(r.market_band),"miss_odds":r.odds,"miss_p_model":r.p_model_bet,"miss_q_market":r.q_bet,"miss_ev":r.ev_bet,"miss_won":False,"control_fixture_id":c.fixture_id,"control_tournament":c.tournament_name,"control_competition_family":c.competition_family,"control_best_of":c.bo_group,"control_market_band":str(c.market_band),"control_odds":c.odds,"control_p_model":c.p_model_bet,"control_q_market":c.q_bet,"control_ev":c.ev_bet,"control_won":True,"matching":"descriptive nearest-confidence winner within family/BO/market band; fallback market band then all winners; not causal"})
    pd.DataFrame(controls).to_csv(output/"severe_misses_matched_wins.csv",index=False)
    tails=[_pb_tail(primary,t) for t in (.8,.9)]
    pd.DataFrame(tails).to_json(output/"favorite_tail_baseline.json",orient="records",indent=2)
    # Monthly complete-universe table and calendar-block intervals for model-market scores and error.
    monthly=primary.groupby("month",dropna=False).agg(n=("fixture_id","size"),expected_favorite_wins=("p_event","sum"),actual_favorite_wins=("y_event","sum"),favorite_probability_error_contribution=("favorite_probability_error_contribution","sum"),excess_log_loss_sum=("excess_log_loss_contribution","sum"),excess_brier_sum=("excess_brier_contribution","sum"),selected_n=("selection_status",lambda x:(x=="selected").sum())).reset_index()
    # Calendar-month blocks include every month in the primary fixture universe.
    monthly.to_csv(output/"monthly_stability.csv",index=False)
    residual=primary.assign(error=primary.p_event-primary.y_event)
    months = sorted(primary.month.unique())
    uncertainty={"favorite_probability_overprediction_month_block_ci95":_month_interval(residual,"error",months),"primary_total_favorite_probability_overprediction":float(residual.error.sum()),"selected_model_error_ci95":_month_interval(selected.assign(error=selected.p_model_bet-selected.won.astype(float)),"error",months),"selected_ev_error_ci95":_month_interval(selected.assign(error=selected.selected_ev_error),"error",months)}
    compare=[]
    for col in ("confidence_band","market_band"):
        for key,g in primary.groupby(col,dropna=False,observed=False):
            for status,h in g.groupby("selection_status"):
                compare.append({"stratifier":col,"band":str(key),"selection_status":status,**_metrics(h)})
    reconciliation={}
    profile_frame=pd.DataFrame(profiles)
    for label in ("month","competition_family","tournament","best_of","confidence","disagreement","market_probability","selection_status"):
        part=profile_frame.loc[profile_frame.profile.eq(label)]
        group_sum=float(part.signed_error_sum.sum())
        total=float((primary.p_event-primary.y_event).sum())
        reconciliation[label]={"group_overprediction_sum":group_sum,"primary_overprediction_sum":total,"difference":group_sum-total,"reconciles":bool(np.isclose(group_sum,total,atol=1e-9))}
        if not reconciliation[label]["reconciles"]:
            raise ValueError(f"Disjoint {label} favorite-overprediction profile does not reproduce primary total")
    selected_checks={}
    for label in ("selected_month","selected_competition_family","selected_tournament","selected_best_of","selected_confidence","selected_disagreement","selected_market_probability","selected_odds","selected_ev"):
        part=profile_frame.loc[profile_frame.profile.eq(label)]
        selected_checks[label]={"group_n":int(part.n.sum()),"selected_n":len(selected),"coverage_complete":int(part.n.sum())==len(selected)}
        if not selected_checks[label]["coverage_complete"]:
            raise ValueError(f"Selected {label} groups do not cover all saved bets")
    pd.DataFrame(compare).to_csv(output/"selection_controls_by_probability_band.csv",index=False)
    plotsets=[("month",monthly.rename(columns={"favorite_probability_error_contribution":"signed_error_sum"}),"Monthly A0 favorite overprediction error"),("competition_family",pd.DataFrame([r for r in profiles if r["profile"]=="competition_family"]),"Favorite probability error by competition family"),("confidence",pd.DataFrame([r for r in profiles if r["profile"]=="confidence"]),"Favorite probability error by A0 confidence band")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name,df,title in plotsets:
        fig,ax=plt.subplots(figsize=(11,5))
        if name=="month": x=df.month.astype(str); vals=df.signed_error_sum; ns=df.n
        else: x=df.group.astype(str); vals=df.signed_error_sum; ns=df.n
        colors=np.where(vals.to_numpy(float)>0,"#b4473c","#397a59")
        bars=ax.bar(x,vals,color=colors); ax.axhline(0,color="black",linewidth=.8); ax.set_ylabel("Sum(A0 favorite probability - favorite win)"); ax.set_title(title+" (labels show N)"); ax.tick_params(axis="x",rotation=60)
        plt.setp(ax.get_xticklabels(), ha="right", rotation=60, fontsize=8)
        for bar,n,v in zip(bars,ns,vals): ax.text(bar.get_x()+bar.get_width()/2,v,f"N={int(n)}",ha="center",va="bottom" if v>=0 else "top",fontsize=7)
        fig.tight_layout(); fig.savefig(output/f"{name}_error_profile.png",dpi=150); plt.close(fig)
    # Name entities only by canonical metadata IDs; no inferred team aliases.
    primary["favorite_team_id"]=np.where(primary.favorite_is_a,primary.origin_team1_id,primary.origin_team2_id)
    primary["favorite_probability"]=primary.p_event
    primary["favorite_won"]=primary.y_event.eq(1)
    primary["favorite_error"]=primary.favorite_probability-primary.favorite_won.astype(float)
    team_rows=[]
    for team,g in primary.groupby("favorite_team_id",dropna=False):
        team_rows.append({"favorite_team_id":str(team),"n_fixtures":len(g),"expected_favorite_wins":float(g.favorite_probability.sum()),"actual_favorite_wins":int(g.favorite_won.sum()),"favorite_overprediction_contribution":float(g.favorite_error.sum()),"favorite_overprediction_per_fixture":float(g.favorite_error.mean()),"favorite_miss_count":int((~g.favorite_won).sum()),"favorite_miss_excess":float((g.favorite_probability*(~g.favorite_won)).sum())})
    team_profile=pd.DataFrame(team_rows).sort_values("favorite_overprediction_contribution",ascending=False)
    team_profile.to_csv(output/"favorite_team_error_profiles.csv",index=False)
    # Chronological split is descriptive sensitivity, not a selected cutoff.
    ordered=sorted(primary.month.dropna().unique())
    split=len(ordered)//2
    primary["temporal_half"]=np.where(primary.month.isin(ordered[:split]),"earlier_months","later_months")
    split_rows=[{"period":k,**_metrics(g)} for k,g in primary.groupby("temporal_half")]
    pd.DataFrame(split_rows).to_csv(output/"temporal_split_sensitivity.csv",index=False)
    full_inventory=opp.loc[opp.cohort.eq("oddspapi_common")&opp.source.eq("best_price")&opp.horizon.eq("T-1h")]
    family_rows=profile_frame.loc[profile_frame.profile.eq("competition_family")]
    tournament_rows=profile_frame.loc[profile_frame.profile.eq("tournament")]
    selected_family_rows=profile_frame.loc[profile_frame.profile.eq("selected_competition_family")]
    selected_month_rows=profile_frame.loc[profile_frame.profile.eq("selected_month")]
    adverse_ll=primary.excess_log_loss_contribution.clip(lower=0)
    conc={"primary_excess_log_loss_sum_vs_market":float(primary.excess_log_loss_contribution.sum()),"primary_excess_brier_sum_vs_market":float(primary.excess_brier_contribution.sum()),"favorite_probability_overprediction_sum":float(primary.favorite_probability_error_contribution.sum()),"top_families_by_excess_log_loss":family_rows.sort_values("excess_log_loss_sum",ascending=False).head(10).to_dict("records"),"top_families_by_excess_brier":family_rows.sort_values("excess_brier_sum",ascending=False).head(10).to_dict("records"),"top_families_by_favorite_overprediction":family_rows.sort_values("signed_error_sum",ascending=False).head(10).to_dict("records"),"top_tournaments_by_excess_log_loss":tournament_rows.sort_values("excess_log_loss_sum",ascending=False).head(15).to_dict("records"),"top_selected_families_by_EV_error":selected_family_rows.sort_values("selected_ev_error_sum").head(10).to_dict("records"),"selected_month_EV_error_contributions":selected_month_rows[["group","n","selected_ev_error_sum","selected_ev_error_per_bet"]].to_dict("records"),"largest_10_fixture_share_of_positive_excess_log_loss":float(adverse_ll.nlargest(10).sum()/adverse_ll.sum()) if adverse_ll.sum() else None,"largest_25_fixture_share_of_positive_excess_log_loss":float(adverse_ll.nlargest(25).sum()/adverse_ll.sum()) if adverse_ll.sum() else None}
    # Source identity and coverage.
    summary={"summary":"Exploratory repaired-prior-roster profile audit; descriptive associations, not causal findings.","cohort":{"primary_definition":"oddspapi_common / best_price / T-1h / available and q25-conflict excluded","n":len(primary),"selected_raw_ev5":len(selected),"current_application_filters":len(current),"primary_unique_fixtures":int(primary.fixture_id.nunique()),"canonical_target_roster_separate_n":len(tables["canonical"]),"canonical_prior_comparable_n":int(canon.canonical_scenario_eligible.sum()),"canonical_same_players_and_cutoff_only":True},"primary_metrics":_metrics(primary),"primary_market_metrics":{**_metrics(primary,"q_consensus_a","y_event")},"selected_metrics":{"n":len(selected),"model":_metrics(pd.DataFrame({"p_event":selected.p_model_bet,"y_event":selected.won})),"market":_metrics(pd.DataFrame({"p_event":selected.q_bet,"y_event":selected.won})),"selected_probability_for_ev_error_sum":float(selected.selected_error.sum()),"selected_probability_for_ev_error_per_bet":float(selected.selected_error.mean()),"sum_realized_minus_saved_ev_pnl_units":float((selected.won-selected.p_for_ev).sum())},"uncertainty":uncertainty,"tail_baseline":tails,"concentration":conc,"matched_controls_n":len(controls),"artifacts":["profiles.csv","primary_fixture_profiles.parquet","selected_fixture_profiles.parquet","fixture_error_contributions.csv","favorite_tail_baseline.json","severe_misses_matched_wins.csv","monthly_stability.csv","selection_controls_by_probability_band.csv","month_error_profile.png","competition_family_error_profile.png","confidence_error_profile.png"],"limitations":["Canonical target-roster cohort (11,550) is a separate retrospective scenario, not pooled with repaired prior-roster primary forecasts.","Competition/tournament profiles are exploratory; names and groupings are supplied by prepared canonical metadata.","Severe-miss matching is descriptive only and does not establish causality.","Heterogeneous-p Poisson-binomial tail probability assumes independent fixtures; shared teams, tournaments, and months create dependence.","Historical roster announcements and quote bindings remain unverified as specified by the frozen protocol."]}
    # Hashes of immutable prepared data are recorded by runner receipt.
    summary["cohort"]["canonical_prior_comparable_n"]=int(primary.canonical_prior_comparable.sum())
    summary["cohort"]["primary_hindsight_roster_change_n"]=int(primary.origin_hindsight_any_roster_change.astype(bool).sum())
    summary["cohort"]["canonical_prior_comparable_scope"]="Joined to the primary 1,852 unique best_price T-1h eligible fixtures only."
    summary["canonical_A_signed_residual_diagnostic_only"]=float(primary.canonical_signed_error.sum())
    summary["selected_metrics"]["mean_saved_raw_EV"]=float(pd.to_numeric(selected.ev_bet).mean())
    summary["selected_metrics"]["mean_realized_after_tax_net_return"]=float(selected.realized_net_after_tax.mean())
    summary["selected_metrics"]["mean_saved_EV_minus_realized_after_tax_return"] = float((pd.to_numeric(selected.ev_bet)-selected.realized_net_after_tax).mean())
    summary["selected_metrics"]["sum_realized_minus_saved_ev_pnl_units"]=float(selected.selected_ev_error.sum())
    summary["concentration"]["top_adverse_favorite_teams"]=team_profile.head(20).to_dict("records")
    summary["concentration"]["most_adverse_family_rows"] = family_rows.sort_values("excess_log_loss_sum",ascending=False).head(10).to_dict("records")
    summary["concentration"]["most_adverse_selected_family_rows"] = selected_family_rows.sort_values("selected_ev_error_sum").head(10).to_dict("records")
    summary["disjoint_profile_reconciliation"] = reconciliation
    summary["selected_profile_coverage"] = selected_checks
    summary["cohort"]["availability_coverage"]={"best_price_t1h_rows":len(full_inventory),"available_non_q25_rows":len(primary),"eligible_share_of_inventory":len(primary)/len(full_inventory) if len(full_inventory) else None}
    summary["selected_metrics"]["selected_model_probability_error_sum"]=float((selected.p_model_bet.astype(float)-selected.won.astype(float)).sum())
    summary["selected_metrics"]["selected_ev_error_sum_realized_net_minus_saved_ev"]=float(selected.selected_ev_error.sum())
    summary["selected_metrics"]["selected_ev_error_mean_per_bet"]=float(selected.selected_ev_error.mean())
    summary["concentration"]["top_adverse_favorite_teams"]=team_profile.head(20).to_dict("records")
    summary["temporal_split_sensitivity"] = split_rows
    summary["artifacts"] += ["favorite_team_error_profiles.csv","temporal_split_sensitivity.csv"]
    summary["input_sha256"]=json.loads((Path(protocol["input_dir"])/"prepared_receipt.json").read_text())["sha256"] if (Path(protocol["input_dir"])/"prepared_receipt.json").exists() else {}
    canonical_reference=tables["canonical"]
    summary["canonical_target_roster_reference_scenario"]={"n":len(canonical_reference),"forecast_and_outcome_scope":"Annual canonical target-roster p_full vs y over all rows, separate from repaired prior-roster primary; market_q is A0-favorite-oriented, not canonical-A, so it is excluded and available market_open is scored in canonical A orientation.","model":_metrics(pd.DataFrame({"p_event":canonical_reference.p_full,"y_event":canonical_reference.y})),"market_open":_metrics(pd.DataFrame({"p_event":canonical_reference.market_open,"y_event":canonical_reference.y}))}
    return summary
