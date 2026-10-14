import importlib.util
import json
import hashlib
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/research/finalize_thesis_audit.py'
spec=importlib.util.spec_from_file_location('thesis_finalizer_test_module',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def setup_evidence(tmp_path):
    main=tmp_path/'main';supp=tmp_path/'supp';main.mkdir();supp.mkdir()
    inputs=[]
    for name in ['features.parquet','odds.csv','identities.json']:
        p=tmp_path/name;p.write_text(name);inputs.append(p)
    selection={'selected':{'logistic':{'C':.1},'mlp':{'alpha':.01}},'primary_family':'mlp'}
    selected=main/'selection.json';selected.write_text(json.dumps(selection))
    protocol={'sources':{str(p):sha(p) for p in inputs}}
    (main/'protocol.json').write_text(json.dumps(protocol))
    supplement={'sources':{str(inputs[0]):sha(inputs[0]),str(selected):sha(selected)},
                'params':selection['selected']['logistic'],'primary_family':'mlp','primary_params':selection['selected']['mlp']}
    (supp/'protocol.json').write_text(json.dumps(supplement))
    for root,name,keys in [(main/'models','logistic',('prediction_sha256','receipt_sha256')),
                           (supp,'no_binomial',('predictions_sha256','receipts_sha256'))]:
        d=root/name;d.mkdir(parents=True);p=d/'predictions.parquet';p.write_text('predictions')
        r=d/'fit_receipts.json';r.write_text('receipts');(d/'complete.json').write_text(json.dumps({keys[0]:sha(p),keys[1]:sha(r)}))
    return main,supp,inputs


def test_preflight_accepts_both_completion_key_schemas(tmp_path):
    main,supp,inputs=setup_evidence(tmp_path)
    receipt=module.verify_evidence(main,supp,*inputs,['logistic'],['no_binomial'])
    assert receipt['status']=='PASS'
    assert len(receipt['streams'])==2


@pytest.mark.parametrize('mutation',['prediction','feature','selection','supplement_params'])
def test_preflight_rejects_changed_evidence(tmp_path,mutation):
    main,supp,inputs=setup_evidence(tmp_path)
    if mutation=='prediction':(main/'models/logistic/predictions.parquet').write_text('changed')
    if mutation=='feature':inputs[0].write_text('changed')
    if mutation=='selection':(main/'selection.json').write_text('{}')
    if mutation=='supplement_params':
        p=supp/'protocol.json';s=json.loads(p.read_text());s['params']={'C':99};p.write_text(json.dumps(s))
    with pytest.raises(ValueError):module.verify_evidence(main,supp,*inputs,['logistic'],['no_binomial'])
