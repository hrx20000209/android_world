"""Replay finds an option chosen by value that starts off-screen (ISSUE #11)."""

from types import SimpleNamespace as NS

from android_world.parallel_exploration import procedure_replay as pr

A = "com.arduia.expense/.ui.MainActivity"


def test_option_off_screen_is_found_by_swiping_and_search_stops_when_list_leaves():
  def chip(label, x):
    return [NS(text="", content_desc="", resource_id="app:id/cd_category", class_name="android.widget.FrameLayout",
               bounds=(x, 780, x + 200, 840), clickable=True),
            NS(text=label, content_desc="", resource_id="app:id/tv_name", class_name="android.widget.TextView",
               bounds=(x + 10, 790, x + 190, 830), clickable=False)]

  def screen(labels, offset):
    els = []
    for i, l in enumerate(labels):
      x = 40 + 220 * i - offset
      if 0 <= x and x + 200 <= 1080:
        els += chip(l, x)
    return NS(activity=NS(component=A), elements=tuple(els))

  cats = ["Food", "Social", "Travel", "Health", "Others"]
  hop = pr.Hop(step=4, source_activity=A, resource_id="app:id/cd_category", class_name="android.widget.FrameLayout",
               own_label="", inner=("Social",), entity="", target_activity=A, target_labels=(), state_ok=True,
               field="category_name")
  state = {"offset": 0}
  def capture(): return screen(cats, state["offset"])
  clicks = []
  def click(a):
    clicks.append(a)
    if a.get("action_type") == "swipe_xy":
      assert 0.1 * 1080 <= a["start"][0] <= 0.9 * 1080 and 0.1 * 1080 <= a["end"][0] <= 0.9 * 1080
      assert a["start"][1] == a["end"][1] == 810  # along the option row
      state["offset"] += 440 if a["start"][0] > a["end"][0] else -440
  plan = pr.Plan("Legal Fees", "Social Club Dues", (hop, hop), {"category_name": "Others"})
  start = capture().elements[0]
  r = pr.execute(plan, start, capture=capture, click=click, sleep=lambda s: None, screen_size=(1080, 2400))
  swipes = [c for c in clicks if c["action_type"] == "swipe_xy"]
  taps = [c for c in clicks if c["action_type"] == "click"]
  assert r.reason == "stopped_before_commit", r
  assert len(swipes) >= 1 and taps[-1]["x"] > 0, clicks
  # The tapped chip is the one labelled Others on the screen after swiping.
  others = [e for e in capture().elements if e.text == "Others"][0]
  assert abs(taps[-1]["x"] - (others.bounds[0] + others.bounds[2]) / 2) <= 100, (taps, others.bounds)

  # A swipe that removes the option family (e.g. switches tab) stops the search.
  state["offset"] = 0
  def click2(a):
    clicks.append(a)
    if a.get("action_type") == "swipe_xy": state["offset"] = 99999
  plan2 = pr.Plan("Legal Fees", "Social Club Dues", (hop, hop), {"category_name": "Others"})
  r2 = pr.execute(plan2, capture().elements[0], capture=capture, click=click2, sleep=lambda s: None, screen_size=(1080, 2400))
  assert r2.reason == "selector_missing_or_ambiguous", r2
