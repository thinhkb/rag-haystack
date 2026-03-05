from rag_haystack.apps.rag_api.security import build_access_predicate

def test_confidentiality_gate():
    pred = build_access_predicate(
        user_context={"confidentiality_level": "internal", "department":"HR", "role":"staff"},
        filters={}
    )
    assert pred({"confidentiality_level":"public", "department":"HR"}) is True
    assert pred({"confidentiality_level":"restricted", "department":"HR"}) is False

def test_department_gate():
    pred = build_access_predicate(
        user_context={"confidentiality_level":"internal", "department":"HR"},
        filters={}
    )
    assert pred({"confidentiality_level":"public", "department":"HR"}) is True
    assert pred({"confidentiality_level":"public", "department":"IT"}) is False