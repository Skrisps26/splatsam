import pytest

from splatsam.authorization import AuthorizationError, require_live
from splatsam.utils import save_json


def test_no_go_means_no_live_action(tmp_path):
    p = tmp_path / "authorization.json"
    with pytest.raises(AuthorizationError):
        require_live("B1", manifest=p)
    save_json(p, {"go": True, "actions": {"B2": True}})
    with pytest.raises(AuthorizationError, match="overlay"):
        require_live("B2", manifest=p)
    save_json(p, {"go": True, "masks_approved": True, "actions": {"B3": True}})
    with pytest.raises(AuthorizationError, match="spend"):
        require_live("B3", manifest=p)
