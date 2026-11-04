"""Video perception: frame sampling, semantic objects, masks, and metric reference."""
from splatsam.perception.frames import FrameSample, sample_frames
from splatsam.perception.objects import identify_objects
from splatsam.perception.reference import ReferenceError, ReferenceResult, find_reference
from splatsam.perception.sam import MockSamClient, SamClient, associate_instances, save_overlays, stability_gate

__all__ = [
    "FrameSample",
    "MockSamClient",
    "ReferenceError",
    "ReferenceResult",
    "SamClient",
    "associate_instances",
    "find_reference",
    "identify_objects",
    "sample_frames",
    "save_overlays",
    "stability_gate",
]
