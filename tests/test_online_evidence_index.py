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
    self.assertEqual(index.retrieve("list", "v1", "meeting"), [])

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


if __name__ == "__main__":
  unittest.main()
