from rag_haystack.libs.ingestion.metadata import normalize_metadata

def test_defaults():
    m = normalize_metadata(None, fallback_doc_id="SOP-1", fallback_title="T")
    assert m["department"] == "UNKNOWN"
    assert m["confidentiality_level"] == "restricted"

def test_conf_validation():
    m = normalize_metadata({"confidentiality_level": "secret"})
    assert m["confidentiality_level"] == "restricted"

def test_roles_string_to_list():
    m = normalize_metadata({"allowed_roles": "staff, manager"})
    assert m["allowed_roles"] == ["staff", "manager"]