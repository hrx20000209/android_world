# Copyright 2026 The android_world Authors.
"""Tests for the grounding guard on generated screen descriptions.

Nothing here loads the 0.6B: `_grounded` is a pure function and the failure it
exists to stop is a content failure, not a runtime one.
"""

from android_world.parallel_exploration.semantic_service import _grounded


# Every case below is a real (activity, labels, reply) triple from
# results/mobileexplorer_runs/model116_oldprompt, where 48% of the 0.6B's
# descriptions named a category no label on the screen supported.


def test_rejects_a_category_the_labels_do_not_support():
  """The dominant failure: a first-run consent dialog called a settings page."""
  assert not _grounded(
      "A settings page with Chrome options.",
      ["Accept & continue", "No thanks", "Yes, I'm in"])


def test_rejects_a_few_shot_answer_echoed_verbatim():
  assert not _grounded("The main settings page.",
                       ["Files that you download appear here", "Downloads"])


def test_rejects_a_media_player_read_onto_an_audio_recorder_setup_screen():
  assert not _grounded("A media player with audio recording options.",
                       ["M4A", "WAV", "3GP", "44.1kHz"])


def test_accepts_a_sentence_that_quotes_a_label():
  assert _grounded('A screen with a "Get started" button.', ["Get started"])


def test_the_package_name_alone_never_grounds_a_sentence():
  """It appears in every sentence about the app, so it separates nothing."""
  assert not _grounded("A chrome page.", ["Accept & continue"])


def test_accepts_a_plural_against_a_singular_label():
  assert _grounded("A list of recordings.", ["Recording"])


def test_a_sentence_of_pure_filler_is_not_grounded():
  """Stopwords must not be able to pass the check on their own."""
  assert not _grounded("The screen shows a list of options.", ["OK"])


def test_short_tokens_do_not_ground_by_accident():
  """Two-letter fragments match almost any haystack as substrings."""
  assert not _grounded("An ad on it.", ["Address"])


# --- the chooser's reply must resolve back to a control on the screen --------
#
# A generated label that is not on screen cannot be probed, and treating a
# near-miss as a hit is how a chooser silently becomes a random ranker.

from android_world.parallel_exploration.semantic_service import _match_candidate
from android_world.parallel_exploration.semantic_service import _choose_prompt


def test_an_exact_reply_resolves_to_that_candidate():
  assert _match_candidate("Search", ["Departure date", "Search"]) == "Search"


def test_quotes_a_trailing_period_and_stray_spacing_are_forgiven():
  assert _match_candidate('  "Search".  ', ["Search", "Cancel"]) == "Search"


def test_case_and_inner_whitespace_are_forgiven():
  assert _match_candidate("network &  INTERNET",
                          ["Network & internet", "Apps"]) == "Network & internet"


def test_a_clipped_reply_resolves_only_when_the_prefix_is_unique():
  assert _match_candidate("Network", ["Network & internet", "Apps"]) == (
      "Network & internet")
  assert _match_candidate("Network", ["Network & internet", "Network mode"]) == ""


def test_a_label_that_is_not_on_the_screen_is_refused():
  assert _match_candidate("Submit", ["Search", "Cancel"]) == ""


def test_an_empty_or_explanatory_reply_is_refused():
  assert _match_candidate("", ["Search", "Cancel"]) == ""
  assert _match_candidate("I would press Search next", ["Search"]) == ""


def test_the_prompt_has_the_specified_shape():
  text = _choose_prompt("Find flights.", "The next unfinished step is to submit.",
                        ["Departure date", "Search"])
  assert text == ("Task: Find flights.\n"
                  "The next unfinished step is to submit.\n\n"
                  "Candidate control labels, in arbitrary order:\n"
                  "- Departure date\n- Search")


def test_the_prompt_omits_an_empty_progress_line():
  text = _choose_prompt("Find flights.", "", ["A", "B"])
  assert text.startswith("Task: Find flights.\n\nCandidate")


def test_a_label_carrying_a_newline_still_resolves():
  """The service echoes a whitespace-normalised label; the caller must find
  it. Un-normalised lookup reads as a refusal, which is indistinguishable
  from the model declining."""
  assert _match_candidate("New name", ["New\nname", "Cancel"]) == "New\nname"
