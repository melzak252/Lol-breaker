"""Ground thesis display names in the same team IDs as the predictions."""
import numpy as np
import pandas as pd


def align_identity_names(metadata, identities):
    m=metadata.copy(); source=identities.copy()
    m['golgg_match_id']=m.golgg_match_id.astype(str)
    source['golgg_match_id']=source.match_id.astype(str)
    source=source[['golgg_match_id','team1_id','team2_id','name_1','name_2','t1_win']]
    source=source.rename(columns={'team1_id':'source_team1_id','team2_id':'source_team2_id'})
    joined=m.merge(source,on='golgg_match_id',how='left',validate='one_to_one')
    same=(joined.team1_id.astype(str)==joined.source_team1_id.astype(str))&(joined.team2_id.astype(str)==joined.source_team2_id.astype(str))
    opposite=(joined.team1_id.astype(str)==joined.source_team2_id.astype(str))&(joined.team2_id.astype(str)==joined.source_team1_id.astype(str))
    if not (same^opposite).all(): raise ValueError('Unresolved or ambiguous source team identity')
    # IDs decide the orientation; outcome consistency can only veto it.
    source_y=joined.t1_win.map({True:1,False:0})
    y=np.where(opposite,1-source_y,source_y)
    if not np.array_equal(y,joined.y_true.to_numpy()): raise ValueError('Source outcome conflict')
    joined['team1_name']=np.where(same,joined.name_1,joined.name_2)
    joined['team2_name']=np.where(same,joined.name_2,joined.name_1)
    joined['source_identity_reversed']=opposite
    return joined.drop(columns=['source_team1_id','source_team2_id','name_1','name_2','t1_win'])
