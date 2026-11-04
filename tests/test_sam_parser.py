"""Regression fixture copied from Meta's public parser README, not a live run.
https://github.com/meta-models/meta-sam/blob/main/python/README.md
"""

import numpy as np
from meta_sam_parser import CompletedOutcome, video_segmentation_format

from splatsam.perception.sam import _paste_parsed_mask

# Official complete cropped-mask example (source frame 320 x 334).
TEXT = "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>\n"


def test_fragmented_official_mask_lands_in_source_coordinates():
    parser = video_segmentation_format().create_parser()
    for start in range(0, len(TEXT), 7):
        parser.push(TEXT[start : start + 7])
    result = parser.finish(CompletedOutcome()).result
    record = next(r for r in result.records if r.kind == "mask")
    image = _paste_parsed_mask(record, 334, 320)
    y, x = np.nonzero(image)
    assert x.min() >= 211 and x.max() < 271 and y.min() >= 228 and y.max() < 255
    assert image[:228].sum() == 0 and image[:, :211].sum() == 0
    assert record.object_id == "0" and record.frame.frame_index == 0


def test_sam_adapter_retains_source_frames_and_avoids_needless_recovery(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    from splatsam.llm.schemas import ObjectInfo, ObjectList
    from splatsam.perception.sam import SamClient

    parser = video_segmentation_format().create_parser()
    parser.push("".join(TEXT.replace("<0f>", f"<{i}f>") for i in range(12)))
    parsed = parser.finish(CompletedOutcome()).result

    class Client:
        def sam_request(self, *args, **kwargs):
            return {"parsed": parsed}

        def generate(self, *args, **kwargs):
            raise AssertionError("successful tracks must not trigger paid refinement")

        def _trajectory(self, **kwargs):
            pass

    # This fixture exercises decoding, not live authorization or transport.
    monkeypatch.setattr("splatsam.authorization.require_live", lambda *args, **kwargs: {})
    video = tmp_path / "synthetic.mp4"
    imageio.mimwrite(video, [np.zeros((334, 320, 3), dtype=np.uint8)] * 12, fps=1, macro_block_size=1)
    objects = ObjectList(objects=[ObjectInfo(id="target", name="object", approx_size_cm=(5, 3, 2))])
    tracks = SamClient(Client(), tmp_path / "output").track(video, objects)
    assert len(tracks) == 1 and tracks[0].stable
    assert [f.frame_index for f in tracks[0].frames] == list(range(12))
    assert {f.centroid for f in tracks[0].frames} == {tracks[0].frames[0].centroid}
