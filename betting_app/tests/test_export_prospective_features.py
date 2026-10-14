"""Pre-start exporter proof with local synthetic historical games and announcements."""
import json
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import pytest
from betting_app.services.current_roster_service import ROLE_ORDER
from betting_app.services.tournament_service import TournamentBracket, BracketMatchNode
from betting_app.tests.test_siamese_research_dataset import match
from betting_app.tests.test_tournament_research_snapshot import protocol
from scripts.tournament_research_snapshot import capture_tournament, replay_tournament


def sources(tmp_path):
    now = datetime.now(timezone.utc)
    source = tmp_path/'source'
    source.mkdir()
    bracket = TournamentBracket(id='synthetic', name='Software fixture', region='LCK', format='single_elimination', teams=['a','b'], matches={'final':BracketMatchNode(id='final',name='Final',round_name='Final',bracket_section='upper',best_of=3,team1='a',team2='b')})
    event = {'event_start_at':(now+timedelta(days=1)).isoformat(),'bracket_announced_at':(now-timedelta(hours=1)).isoformat(),'evidence_files':['announcement.txt'],'bracket_sources':['announcement.txt'],'rosters':{team:{'players':dict(zip(ROLE_ORDER,[str(i) for i in range(offset,offset+5)])),'announced_at':(now-timedelta(hours=1)).isoformat(),'source_files':['announcement.txt']} for team,offset in [('a',0),('b',5)]}}
    (source/'event.json').write_text(json.dumps(event))
    (source/'bracket.json').write_text(json.dumps(asdict(bracket)))
    (source/'announcement.txt').write_text('Synthetic software fixture only, not a real announcement.')
    history = tmp_path/'history.json'
    history.write_text(json.dumps([match(i,(now-timedelta(days=4-i)).date().isoformat(),winner=i%2==1) for i in (1,2)]))
    return history, source


def test_exported_features_reconstruct_through_existing_frozen_capture(tmp_path):
    from scripts.export_prospective_features import export_features
    history, source = sources(tmp_path)
    result = export_features(history,source,tmp_path/'export')
    locked = protocol(tmp_path)
    capture_tournament(locked,tmp_path/'export/event_inputs',tmp_path/'capture')
    replay = replay_tournament(locked,tmp_path/'capture',tmp_path/'replay',simulations=10000,seed=89)
    assert replay['models']['outcome']['champion_prob']['a'] == pytest.approx(.5,abs=.02)
    assert result['history']['used_series'] == 2
    assert result['eligible_for_frozen_confirmation'] is False
    with pytest.raises(FileExistsError):
        export_features(history,source,tmp_path/'export')


def test_future_roster_announcement_cannot_be_exported(tmp_path):
    from scripts.export_prospective_features import export_features
    history, source = sources(tmp_path)
    event = json.loads((source/'event.json').read_text())
    event['rosters']['a']['announced_at'] = (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    (source/'event.json').write_text(json.dumps(event))
    with pytest.raises(ValueError):
        export_features(history,source,tmp_path/'export')


@pytest.mark.parametrize('text',['[{"match_id":"1"},]','[{"match_id":"1"}] garbage','[{"match_id":"1"}'])
def test_stream_reader_rejects_truncated_or_malformed_history(tmp_path,text):
    from scripts.export_prospective_features import iter_matches
    path = tmp_path/'history.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_matches(path))


def test_export_replays_sources_and_detects_history_tampering(tmp_path):
    from scripts.export_prospective_features import export_features, verify_export
    history, source = sources(tmp_path)
    export_features(history,source,tmp_path/'export')
    assert verify_export(tmp_path/'export')['verified'] is True
    archive = tmp_path/'export/source/history.json'
    changed = json.loads(archive.read_text())
    changed[0]['games'][0]['t1_stats']['gold'] += 1
    archive.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        verify_export(tmp_path/'export')


def test_new_feature_contract_cannot_enter_legacy_frozen_confirmation(tmp_path):
    from scripts.export_prospective_features import export_features
    from scripts.confirm_oddsfree_student import capture_forecasts
    history, source = sources(tmp_path)
    event = json.loads((source/'event.json').read_text())
    (source/'fixtures.csv').write_text('golgg_match_id,team_a,team_b,best_of,match_start_at\nfixture-1,b,a,3,'+event['event_start_at']+'\n')
    export_features(history,source,tmp_path/'export')
    locked = protocol(tmp_path)
    with pytest.raises(ValueError):
        capture_forecasts(locked,tmp_path/'export/sports.csv',tmp_path/'export/metadata.csv',tmp_path/'forecasts')


def test_stream_reader_preserves_records_across_chunk_boundaries(tmp_path):
    from scripts.export_prospective_features import iter_matches
    records = [{'match_id':'1','text':'ż'*70000}, {'match_id':'2','nested':{'a':[1,2,3]}}]
    path = tmp_path/'history.json'
    path.write_text(json.dumps(records,ensure_ascii=False))
    assert list(iter_matches(path)) == records


def test_stream_reader_detects_gzip_even_after_archive_renaming(tmp_path):
    import gzip
    from scripts.export_prospective_features import iter_matches
    records = [{"match_id": "1", "text": "ż" * 70000}, {"match_id": "2"}]
    path = tmp_path / "history.json"
    path.write_bytes(gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8-sig")))
    assert list(iter_matches(path)) == records


def test_fixture_archive_reconstructs_across_process_hash_seeds(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    history, source = sources(tmp_path)
    event = json.loads((source/'event.json').read_text())
    (source/'fixtures.csv').write_text('golgg_match_id,team_a,team_b,best_of,match_start_at\nfixture-1,b,a,3,'+event['event_start_at']+'\n')
    root = Path(__file__).resolve().parents[2]
    commands = [
        ['export','--history-json',str(history),'--event-dir',str(source),'--output-dir',str(tmp_path/'export')],
        ['verify','--export-dir',str(tmp_path/'export')],
    ]
    for seed, arguments in enumerate(commands):
        result = subprocess.run([sys.executable,'scripts/export_prospective_features.py',*arguments],cwd=root,env={**os.environ,'PYTHONHASHSEED':str(seed)},capture_output=True,text=True)
        assert result.returncode == 0, result.stderr


def test_one_seeded_match_cannot_be_exported_as_two_fixture_ids(tmp_path):
    from scripts.export_prospective_features import export_features
    history, source = sources(tmp_path)
    event = json.loads((source/'event.json').read_text())
    start = event['event_start_at']
    (source/'fixtures.csv').write_text(f'golgg_match_id,team_a,team_b,best_of,match_start_at\nfirst,a,b,3,{start}\nsecond,b,a,3,{start}\n')
    with pytest.raises(ValueError):
        export_features(history,source,tmp_path/'export')
