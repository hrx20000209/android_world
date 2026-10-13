import unittest
from android_world.parallel_exploration.online_evidence_index import OnlineEvidenceIndex


class EvidenceLifecycleTest(unittest.TestCase):
  def test_uncommitted_stale_and_failed_recovery_are_not_consumed(self):
    index = OnlineEvidenceIndex()
    index.add(source="list", source_version="v1", destination="detail",
              action={"control_key": "note"}, labels=["meeting attendee count"],
              recovered=True, cost_s=1)
    self.assertEqual(index.retrieve("list", "v1", "meeting"), [])
    index.commit()
    self.assertEqual(len(index.retrieve("list", "v1", "meeting")), 1)
    self.assertEqual(index.retrieve("list", "v2", "meeting"), [])
    index.add(source="list", source_version="v1", destination="detail",
              action={"control_key": "note"}, labels=["meeting"],
              recovered=False, cost_s=1)
    index.commit()
    # A failed later retry cannot invalidate the earlier successful probe.
    self.assertEqual(len(index.retrieve("list", "v1", "meeting")), 1)

  def test_replacement_removes_old_postings(self):
    index = OnlineEvidenceIndex()
    args = dict(source="s", source_version="v", destination="d",
                action={"control_key": "button"}, recovered=True, cost_s=1)
    index.add(**args, labels=["old"])
    index.commit()
    index.add(**args, labels=["new"])
    index.commit()
    self.assertEqual(index.retrieve("s", "v", "old"), [])
    self.assertEqual(len(index.retrieve("s", "v", "new")), 1)

  def test_same_action_keeps_multiple_verified_destinations(self):
    index = OnlineEvidenceIndex()
    action = {
        "action_type": "CLICK",
        "control_key": "com.example:id/menu|Menu|android.widget.Button",
        "stable_control_key": "resource:com.example:id/menu|class:android.widget.Button",
        "x": 300,
        "y": 200,
    }
    for destination, labels in (("settings-v1", ["Settings"]), ("search-v1", ["Search"])):
      index.add(source="home", source_version="layout-v1", destination=destination,
                destination_family="menu-page", action=action, labels=labels,
                recovered=True, cost_s=.2)
    index.commit()

    self.assertEqual(len(index.records), 2)
    self.assertEqual(
        index.outcomes[("home", "layout-v1", index.classify_transition(
            source="home", source_version="layout-v1", action=action,
            destination="settings-v1")["action_key"])],
        {"settings-v1": 1, "search-v1": 1},
    )
    self.assertEqual(len(index.retrieve("home", "layout-v1", "settings search")), 2)
    self.assertEqual(
        index.classify_transition(
            source="home", source_version="layout-v1", action=action,
            destination="settings-v2", destination_family="menu-page",
        )["same_family_prior_destinations"],
        ["search-v1", "settings-v1"],
    )

  def test_classifier_distinguishes_modal_known_alternative_new_and_unmatchable(self):
    index = OnlineEvidenceIndex()
    action = {"action_type": "click", "stable_control_key": "resource:menu"}
    for destination in ("settings", "settings", "search"):
      index.add(source="home", source_version="v1", destination=destination,
                action=action, labels=[destination], recovered=True, cost_s=.1)

    def classify(destination, **kwargs):
      return index.classify_transition(
          source="home", source_version="v1", action=action,
          destination=destination, **kwargs,
      )["verdict"]

    self.assertEqual(classify("settings"), "KNOWN_OUTCOME")
    self.assertEqual(classify("search"), "KNOWN_ALTERNATIVE")
    self.assertEqual(classify("help"), "NEW_BRANCH")
    self.assertEqual(classify("settings", action_match=False), "UNMATCHABLE")

  def test_failed_probe_is_not_a_known_destination(self):
    index = OnlineEvidenceIndex()
    action = {"action_type": "click", "stable_control_key": "resource:menu"}
    index.add(source="home", source_version="v1", destination="settings",
              action=action, labels=["Settings"], recovered=False, cost_s=.4)

    result = index.classify_transition(
        source="home", source_version="v1", action=action,
        destination="settings",
    )
    self.assertEqual(result["verdict"], "UNMATCHABLE")
    self.assertEqual(result["reason"], "no_verified_prior_edge")
    index.commit()
    self.assertEqual(index.retrieve("home", "v1", "settings"), [])

  def test_later_failed_duplicate_does_not_erase_verified_edge(self):
    index = OnlineEvidenceIndex()
    action = {"action_type": "click", "stable_control_key": "resource:menu"}
    args = dict(source="home", source_version="v1", destination="settings",
                action=action, labels=["Settings"], cost_s=.1)
    index.add(**args, recovered=True)
    index.add(**args, recovered=False)
    index.commit()

    self.assertEqual(index.events[-1]["event"], "rejected_duplicate")
    self.assertEqual(len(index.retrieve("home", "v1", "settings")), 1)

  def test_action_type_is_part_of_transition_identity(self):
    index = OnlineEvidenceIndex()
    control = {"stable_control_key": "resource:row"}
    index.add(source="list", source_version="v1", destination="detail",
              action={**control, "action_type": "click"}, labels=["Detail"],
              recovered=True, cost_s=.1)

    result = index.classify_transition(
        source="list", source_version="v1",
        action={**control, "action_type": "long_press"}, destination="detail",
    )
    self.assertEqual(result["verdict"], "UNMATCHABLE")


if __name__ == "__main__":
  unittest.main()
