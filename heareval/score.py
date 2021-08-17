"""
Common utils for scoring.
"""
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import sed_eval
from dcase_util.containers import MetaDataContainer


def label_vocab_as_dict(df: pd.DataFrame, key: str, value: str) -> Dict:
    """
    Returns a dictionary of the label vocabulary mapping the label column to
    the idx column. key sets whether the label or idx is the key in the dict. The
    other column will be the value.
    """
    if key == "label":
        # Make sure the key is a string
        df["label"] = df["label"].astype(str)
        value = "idx"
    else:
        assert key == "idx", "key argument must be either 'label' or 'idx'"
        value = "label"
    return df.set_index(key).to_dict()[value]


class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    # TODO: Remove task_metadata since we don't use it much
    def __init__(
        self,
        task_metadata: Dict,
        label_to_idx: Dict[str, int],
        name: Optional[str] = None,
    ):
        self.task_metadata = task_metadata
        self.label_to_idx = label_to_idx
        if name:
            self.name = name

    def __call__(self, predictions: Any, targets: Any, **kwargs) -> float:
        """
        Compute the score based on the predictions and targets. Returns the score.
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name


class Top1Error(ScoreFunction):
    name = "top1_err"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            if predicted_class == target_class:
                correct += 1

        return correct / len(targets)


class ChromaError(ScoreFunction):
    """
    Score specifically for pitch detection -- converts all pitches to chroma first.
    This score ignores octave errors in pitch classification.
    """

    name = "chroma_err"

    def __call__(self, predictions: np.ndarray, targets: List, **kwargs) -> float:
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            # Ignore octave errors by converting the predicted class to chroma before
            # checking for correctness.
            if predicted_class % 12 == target_class % 12:
                correct += 1

        return correct / len(targets)


class SoundEventScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval
    """

    # Score class must be defined in inheriting classes
    score_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
        self, task_metadata: Dict, label_to_idx: Dict[str, int], params: Dict = None
    ):
        super().__init__(task_metadata=task_metadata, label_to_idx=label_to_idx)
        self.params = params if params is not None else {}
        assert self.score_class is not None

    def __call__(self, predictions: Dict, targets: Dict, **kwargs):
        # Containers of events for sed_eval
        reference_event_list = self.sed_eval_event_container(targets)
        estimated_event_list = self.sed_eval_event_container(predictions)

        # This will break in Python < 3.6 if the dict order is not
        # the insertion order I think. I'm a little worried about this line
        scores = self.score_class(
            event_label_list=list(self.label_to_idx.keys()), **self.params
        )

        for filename in predictions:
            scores.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        # This (and segment_based_scores) return a pretty large selection of scores.
        # We might want to add a task_metadata option to filter these for the specific
        # score that we are going to use to evaluate the task.
        overall_scores = scores.results_overall_scores()
        return overall_scores

    @staticmethod
    def sed_eval_event_container(x: Dict) -> MetaDataContainer:
        # Reformat event list for sed_eval
        reference_events = []
        for filename, event_list in x.items():
            for event in event_list:
                reference_events.append(
                    {
                        # Convert from ms to seconds for sed_eval
                        "event_label": event["label"],
                        "event_onset": event["start"] / 1000.0,
                        "event_offset": event["end"] / 1000.0,
                        "file": filename,
                    }
                )
        return MetaDataContainer(reference_events)


class SegmentBasedScore(SoundEventScore):
    """
    segment-based scores - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.SegmentBasedMetrics


class EventBasedScore(SoundEventScore):
    """
    event-based scores - the ground truth and system output are compared at
    event instance level;

    See https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedScores.html # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.EventBasedMetrics


available_scores: Dict[str, Callable] = {
    "top1_err": Top1Error,
    "pitch_err": partial(Top1Error, name="pitch_err"),
    "chroma_err": ChromaError,
    "onset_only_event_based": partial(
        EventBasedScore, params={"evaluate_offset": False, "t_collar": 0.2}
    ),
    "segment_based": SegmentBasedScore,
}