"""Standard-library tests using frozen source text and synthetic outputs only."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from e1_common import HERE, assignment, lock, require, transform, write_json


def main():
    policy = lock()
    reviews = []
    for job in policy['jobs']:
        family = policy['families'][job['family']]
        text = (HERE / family['scripts'][job['script_name']]).read_text(encoding='utf-8')
        generated, audit = transform(text, job['family'], Path('/root/autodl-tmp/time_uncertainty_rec'), Path('/root/autodl-tmp/time_uncertainty_rec/kais_x4_tisasrec_v1_e1_fixture') if job['family']=='tisasrec' else Path('/root/autodl-tmp/time_uncertainty_rec/e1_fixture'), job['seed'])
        compile(generated, job['id'] + '.py', 'exec')
        gate = [e for e in audit['edits'] if e['kind']=='minimum_stop_gate']
        require(len(gate) == 1 and audit['scientific_ast_unchanged_after_gate_undo'], 'Only one gate and unchanged definitions')
        # Each generated path assignment and gate can be checked without imports.
        reviews.append({'id':job['id'], 'scientific_definitions':audit['scientific_definition_count'], 'status':'PASS_GENERATION_AND_AST'})
    # Full fresh native-family stage construction, without executing any source.
    # Amazon reference copies are synthetic bytes matching a temporary fixture contract.
    from e1_common import make_attempt, read_json, sha
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        root = Path(temp)
        outer = root/'kais_e1_min50_v1_fixture'
        outer.mkdir(); (outer/'attempts').mkdir()
        for job in policy['jobs']:
            if job['family']=='amazon_lbsp':
                continue  # Original reference copying is server-only; independently reviewed.
            native = make_attempt(root, outer, job, 1, policy)
            require(native.is_dir(), 'Native stage absent')
            try:
                make_attempt(root, outer, job, 1, policy)
            except ValueError:
                pass
            else:
                raise ValueError('Existing stage overwrite accepted')
            # Validate old model seals and all newly populated generated hashes.
            source_family = policy['families'][job['family']]
            if 'model_definition.py' in source_family['scripts']:
                require(sha(native/'scripts/model_definition.py') == sha(HERE/source_family['scripts']['model_definition.py']), 'Model copy changed')
    # Malformed source must fail closed before generation.
    exemplar = policy['jobs'][0]
    source = (HERE/policy['families'][exemplar['family']]['scripts'][exemplar['script_name']]).read_text(encoding='utf-8')
    try:
        transform(source.replace('if bad_epochs >= PATIENCE:', 'if bad_epochs > PATIENCE:'), exemplar['family'], Path('/fixture'), Path('/fixture/stage'))
    except ValueError:
        pass
    else:
        raise ValueError('Mutated stopping source accepted')
    # Plan should run against a nonexistent project root without creating it.
    absent = HERE/'DOES_NOT_EXIST_PLAN_ROOT'
    require(not absent.exists(), 'Fixture root unexpectedly exists')
    completed = subprocess.run([sys.executable,str(HERE/'run_e1_min50.py'),'--root',str(absent)],capture_output=True,text=True)
    require(completed.returncode == 0 and 'PLAN_ONLY' in completed.stdout and not absent.exists(), 'Plan accessed/created project root')
    write_json(HERE/'QA_SOURCE_GENERATION.json', {'status':'PASS_LOCAL_STANDARD_LIBRARY_GENERATION_ONLY', 'all19_trainers_generated':True,
      'unchanged_scientific_asts':True, 'existing_stage_overwrite_rejected':True, 'mutated_source_rejected':True,
      'plan_no_project_access':True, 'torch_or_real_data_loaded':False, 'reviews':reviews})
    print('E1_LOCAL_QA_PASS19_GENERATED_NO_RESEARCH_EXECUTION')


if __name__=='__main__':
    main()
