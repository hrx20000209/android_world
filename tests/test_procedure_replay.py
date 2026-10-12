"""Adversarial boundaries for entity-parameterised procedure replay."""

import dataclasses
from types import SimpleNamespace as NS

from android_world.parallel_exploration import procedure_replay as pr

GOAL = "Delete the following recipes from Broccoli app: Pesto, Garlic Shrimp, Lentil Soup."
ENTITIES = ("Pesto", "Garlic Shrimp", "Lentil Soup")
LIST, DETAIL, MENU, DIALOG = "app/.List", "app/.Detail", "app/.Detail", "app/.Detail"


def el(text="", desc="", rid="", cls="android.widget.LinearLayout", bounds=(0, 0, 10, 10),
       clickable=False):
  return NS(text=text, content_desc=desc, resource_id=rid, class_name=cls, bounds=bounds,
            clickable=clickable)


def row(name, y):
  return [el(rid="", cls="android.widget.FrameLayout", bounds=(0, y, 1000, y + 100), clickable=True),
          el(text=name, cls="android.widget.TextView", bounds=(10, y + 10, 500, y + 60))]


def screen(activity, *elements):
  flat = []
  for item in elements:
    flat.extend(item if isinstance(item, list) else [item])
  return NS(activity=NS(component=activity), elements=tuple(flat))


def list_screen(names=ENTITIES):
  return screen(LIST, *[row(n, 100 + 120 * i) for i, n in enumerate(names)])


def detail(name):
  return screen(DETAIL, el(text=name),
                el(desc="More options", rid="", cls="android.widget.ImageView",
                   bounds=(900, 0, 1000, 80), clickable=True))


def menu(name):
  return screen(MENU, el(text=name), el(cls="android.widget.LinearLayout", bounds=(500, 100, 1000, 200),
                                        clickable=True),
                el(text="Delete", cls="android.widget.TextView", bounds=(510, 110, 990, 190)))


def demonstrate(graph, name="Pesto", commit_ok=False, menu_ok=True):
  lst = list_screen()
  first = next(e for e in lst.elements if e.clickable and e.bounds[1] == 100 + 120 * ENTITIES.index(name))
  graph.record(pr.make_hop(0, lst, first, detail(name), ENTITIES, True))
  d = detail(name)
  graph.record(pr.make_hop(1, d, d.elements[1], menu(name), ENTITIES, True))
  m = menu(name)
  graph.record(pr.make_hop(2, m, m.elements[1], screen(DIALOG, el(text="DELETE")), ENTITIES, menu_ok))
  dialog = screen(DIALOG, el(text="DELETE", rid="android:id/button1", cls="android.widget.Button",
                             bounds=(700, 900, 900, 1000), clickable=True))
  graph.record(pr.make_hop(3, dialog, dialog.elements[0], list_screen(ENTITIES[1:]), ENTITIES, commit_ok))


def test_obligations_must_be_verbatim_in_goal():
  out = '```json\n{"items": ["Pesto", "Garlic Shrimp", "Tofu Curry", "Lentil Soup"]}\n```'
  assert pr.parse_obligations(out, GOAL) == ENTITIES
  assert pr.parse_obligations("I think Pesto", GOAL) == ()


def test_template_is_the_navigation_prefix_before_the_commit():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  template = graph.template()
  assert [h.step for h in template] == [0, 1, 2]
  assert template[0].entity == "Pesto"
  planned, element = graph.plan(list_screen(ENTITIES[1:]))
  assert planned.entity == "Garlic Shrimp" and element.bounds[1] == 100
  # The next entity in task order is not on screen: never skip ahead.
  assert graph.plan(list_screen(("Lentil Soup",))) is None


def test_incomplete_demonstration_is_not_a_template():
  graph = pr.ProcedureGraph(ENTITIES)
  lst = list_screen()
  graph.record(pr.make_hop(0, lst, lst.elements[0], detail("Pesto"), ENTITIES, True))
  assert graph.template() is None and graph.last_rejection == "demonstration_not_completed"


def test_uncertified_hop_blocks_strict_but_not_ablation():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph, menu_ok=None)
  assert graph.template() is None and graph.last_rejection == "uncertified_hop"
  loose = pr.ProcedureGraph(ENTITIES, allow_unverified=True)
  demonstrate(loose, menu_ok=None)
  assert len(loose.template()) == 3


def test_typing_or_answer_breaks_the_procedure():
  graph = pr.ProcedureGraph(ENTITIES)
  lst = list_screen()
  graph.record(pr.make_hop(0, lst, lst.elements[0], detail("Pesto"), ENTITIES, True))
  graph.record(pr.barrier(1))
  assert graph.template() is None


def test_ambiguous_or_missing_entity_refuses():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  assert graph.plan(list_screen(("Garlic Shrimp", "Garlic Shrimp"))) is None
  assert graph.last_rejection == "entity_ambiguous"
  assert graph.plan(list_screen(("Pesto",))) is None
  assert graph.last_rejection == "entity_not_visible"
  assert graph.plan(detail("Garlic Shrimp")) is None
  assert graph.plan(screen("other/.Activity")) is None
  assert graph.last_rejection == "not_on_template_source"


def test_replay_stops_before_commit_and_checks_entity_landing():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  planned, element = graph.plan(list_screen(ENTITIES[1:]))
  screens = iter([detail("Garlic Shrimp"), detail("Garlic Shrimp"), menu("Garlic Shrimp"),
                  menu("Garlic Shrimp"), screen(DIALOG, el(text="DELETE"))])
  clicks = []
  result = pr.execute(planned, element, capture=lambda: next(screens), click=clicks.append, sleep=lambda _: None)
  assert result.reason == "stopped_before_commit" and result.completed == 3 and len(clicks) == 3
  # Wrong entity opened: stop after the first real press, never continue.
  screens = iter([detail("Lentil Soup")])
  clicks = []
  result = pr.execute(planned, element, capture=lambda: next(screens), click=clicks.append, sleep=lambda _: None)
  assert result.reason == "landing_entity_mismatch" and len(clicks) == 1 and result.completed == 0


def test_fixed_hop_requires_unique_named_control():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  hop = graph.template()[1]
  d = detail("Garlic Shrimp")
  assert pr.relocate_fixed(hop, d) is d.elements[1]
  twice = screen(DETAIL, d.elements[1], el(desc="More options", cls="android.widget.ImageView",
                                           bounds=(0, 0, 50, 50), clickable=True))
  assert pr.relocate_fixed(hop, twice) is None


def test_unlabelled_icon_is_named_by_unique_resource_id_only():
  hop = pr.Hop(1, DETAIL, "app:id/btn_delete", "android.widget.ImageView", "", (), "",
               DETAIL, (), True)
  one = screen(DETAIL, el(rid="app:id/btn_delete", cls="android.widget.ImageView", clickable=True))
  assert pr.relocate_fixed(hop, one) is one.elements[0]
  per_row = screen(DETAIL, *[el(rid="app:id/btn_delete", cls="android.widget.ImageView",
                                bounds=(0, y, 10, y + 10), clickable=True) for y in (0, 50)])
  assert pr.relocate_fixed(hop, per_row) is None
  bare = pr.Hop(1, DETAIL, "", "android.widget.ImageView", "", (), "", DETAIL, (), True)
  graph = pr.ProcedureGraph(ENTITIES)
  lst = list_screen()
  graph.record(pr.make_hop(0, lst, lst.elements[0], detail("Pesto"), ENTITIES, True))
  graph.record(bare)
  graph.record(pr.Hop(2, DETAIL, "android:id/button1", "android.widget.Button", "OK", ("OK",), "",
                      LIST, (), False))
  assert graph.template() is None and graph.last_rejection == "unnamed_fixed_hop"


def test_lazy_contract_rebinds_recorded_hops_and_marks_done():
  graph = pr.ProcedureGraph(())
  demonstrate(graph)
  graph.set_entities(())  # as recorded before any contract existed
  assert graph.template() is None and not graph.done
  assert graph.has_completed_demonstration_candidate()
  graph.set_entities(ENTITIES)
  assert graph.template()[0].entity == "Pesto"
  assert graph.done == {"pesto"}


def test_read_only_episode_never_requests_a_contract():
  graph = pr.ProcedureGraph(())
  lst = list_screen()
  graph.record(pr.make_hop(0, lst, lst.elements[0], detail("Pesto"), (), True))
  graph.record(pr.barrier(1))  # answer
  assert not graph.has_completed_demonstration_candidate()


def test_long_press_is_replayed_as_long_press():
  graph = pr.ProcedureGraph(ENTITIES)
  lst = list_screen()
  graph.record(pr.make_hop(0, lst, lst.elements[0], lst, ENTITIES, True, action_type="long_press"))
  dialog = screen(LIST, el(text="CONFIRM", rid="app:id/btn_confirm", cls="android.widget.Button",
                           bounds=(0, 900, 100, 1000), clickable=True))
  graph.record(pr.make_hop(1, dialog, dialog.elements[0], lst, ENTITIES, False))
  planned, element = graph.plan(list_screen(ENTITIES[1:]))
  clicks = []
  pr.execute(planned, element, capture=lambda: list_screen(ENTITIES[1:]), click=clicks.append)
  assert clicks[0]["action_type"] == "long_press"


def test_search_demonstration_supplies_the_entity_by_typing():
  graph = pr.ProcedureGraph(ENTITIES)
  search_btn = el(desc="Search", rid="app:id/search", cls="android.widget.ImageView",
                  bounds=(800, 0, 900, 80), clickable=True)
  field = el(rid="app:id/query", cls="android.widget.EditText", bounds=(0, 0, 700, 80), clickable=True)
  lst = screen(LIST, search_btn, *[row(n, 100 + 120 * i) for i, n in enumerate(("Other", "Thing"))])
  searching = screen(LIST, field)
  filtered = screen(LIST, field, row("Pesto", 100))
  graph.record(pr.make_hop(0, lst, search_btn, searching, ENTITIES, True))
  graph.record(pr.make_typed_hop(1, searching, field, "Pesto", filtered, ENTITIES, True))
  graph.record(pr.make_hop(2, filtered, filtered.elements[1], detail("Pesto"), ENTITIES, True))
  d = detail("Pesto")
  graph.record(pr.make_hop(3, d, d.elements[1], menu("Pesto"), ENTITIES, True))
  m = menu("Pesto")
  graph.record(pr.make_hop(4, m, m.elements[1], screen(DIALOG, el(text="DELETE")), ENTITIES, True))
  dlg = screen(DIALOG, el(text="DELETE", rid="android:id/button1", cls="android.widget.Button",
                          bounds=(700, 900, 900, 1000), clickable=True))
  graph.record(pr.make_hop(5, dlg, dlg.elements[0], lst, ENTITIES, False))
  planned, element = graph.plan(lst)  # Garlic Shrimp is not on screen
  assert planned.entity == "Garlic Shrimp" and element is search_btn
  assert [h.action_type for h in planned.hops] == ["click", "input_text", "click", "click", "click"]
  screens = iter([searching, searching, filtered_for := screen(LIST, field, row("Garlic Shrimp", 100)),
                  filtered_for, detail("Garlic Shrimp"), detail("Garlic Shrimp"), menu("Garlic Shrimp"),
                  menu("Garlic Shrimp"), dlg])
  acts = []
  result = pr.execute(planned, element, capture=lambda: next(screens), click=acts.append, sleep=lambda _: None)
  assert result.reason == "stopped_before_commit" and result.completed == 5
  assert acts[1]["action_type"] == "input_text" and acts[1]["text"] == "Garlic Shrimp"


def test_typing_other_text_is_not_a_slot():
  field = el(rid="app:id/body", cls="android.widget.EditText", clickable=True)
  hop = pr.make_typed_hop(0, screen(DETAIL, field), field, "Hello world", screen(DETAIL), ENTITIES, True)
  assert hop.entity == ""


def test_typed_content_ends_the_demonstration():
  graph = pr.ProcedureGraph(ENTITIES)
  field = el(rid="app:id/body", cls="android.widget.EditText", bounds=(0, 0, 700, 80), clickable=True)
  lst = list_screen()
  graph.record(pr.make_hop(0, lst, lst.elements[0], detail("Pesto"), ENTITIES, True))
  ed = screen(DETAIL, field)
  graph.record(pr.make_typed_hop(1, ed, field, "Hello world", ed, ENTITIES, True))
  save = el(text="Save", rid="app:id/save", cls="android.widget.Button", bounds=(0, 900, 100, 1000),
            clickable=True)
  graph.record(pr.make_hop(2, screen(DETAIL, save), save, lst, ENTITIES, False))
  # The run before the commit is just the typed content: no entity, no template.
  assert graph.template() is None


def test_replay_starts_at_the_latest_feasible_hop():
  graph = pr.ProcedureGraph(ENTITIES)
  search_btn = el(desc="Search", rid="app:id/search", cls="android.widget.ImageView",
                  bounds=(800, 0, 900, 80), clickable=True)
  field = el(rid="app:id/query", cls="android.widget.AutoCompleteTextView", bounds=(0, 0, 700, 80),
             clickable=True)
  lst = screen(LIST, search_btn)
  searching = screen(LIST, field)
  filtered = screen(LIST, field, row("Pesto", 100))
  graph.record(pr.make_hop(0, lst, search_btn, searching, ENTITIES, True))
  graph.record(pr.make_typed_hop(1, searching, field, "Pesto", filtered, ENTITIES, True))
  graph.record(pr.make_hop(2, filtered, filtered.elements[1], detail("Pesto"), ENTITIES, True))
  dlg = screen(DIALOG, el(text="DELETE", rid="android:id/button1", cls="android.widget.Button",
                          bounds=(700, 900, 900, 1000), clickable=True))
  graph.record(pr.make_hop(3, dlg, dlg.elements[0], searching, ENTITIES, False))
  planned, element = graph.plan(searching)  # search bar still open: start at typing
  assert element is field and planned.hops[0].action_type == "input_text"
  planned, element = graph.plan(lst)
  assert element is search_btn and len(planned.hops) == 3


def test_failed_entity_is_not_retried_and_not_done():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  graph.mark_failed("Garlic Shrimp")
  planned, _ = graph.plan(list_screen(ENTITIES[1:]))
  assert planned.entity == "Lentil Soup"  # a failed entity is the model's, not a blocker
  assert "garlic shrimp" not in graph.done


def test_replayed_typing_clears_the_field_first():
  hop = pr.Hop(0, LIST, "app:id/q", "android.widget.EditText", "", ("Pesto",), "Pesto", LIST, (), True,
               action_type="input_text")
  field = el(rid="app:id/q", cls="android.widget.EditText", clickable=True)
  acts = []
  pr.execute(pr.Plan("Lentil Soup", "Pesto", (hop,)), field, capture=lambda: screen(LIST),
             click=acts.append)
  assert acts[0]["clear_text"] is True and acts[0]["text"] == "Lentil Soup"


def test_control_still_sliding_in_is_retried_not_tapped():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  planned, element = graph.plan(list_screen(ENTITIES[1:]))
  sliding = screen(DETAIL, el(text="Garlic Shrimp", bounds=(1200, 0, 1400, 50)),
                   el(desc="More options", cls="android.widget.ImageView", bounds=(1100, 0, 1200, 80),
                      clickable=True))
  settled = detail("Garlic Shrimp")
  screens = iter([settled, sliding, settled, menu("Garlic Shrimp"), menu("Garlic Shrimp"),
                  screen(DIALOG, el(text="DELETE"))])
  clicks = []
  result = pr.execute(planned, element, capture=lambda: next(screens), click=clicks.append,
                      sleep=lambda _: None, screen_size=(1080, 2400))
  assert result.reason == "stopped_before_commit"
  assert all(c["x"] <= 1080 for c in clicks)


def test_unreplayable_lead_in_is_trimmed_not_fatal():
  graph = pr.ProcedureGraph(ENTITIES)
  lst = list_screen()
  junk = el(cls="android.widget.ImageView", bounds=(0, 0, 50, 50), clickable=True)
  graph.record(pr.make_hop(0, screen(LIST, junk), junk, lst, ENTITIES, True))  # unnamed lead-in
  demonstrate(graph)
  assert [h.step for h in graph.template()] == [0, 1, 2]
  assert graph.template()[0].entity == "Pesto"


def test_unreplayable_hop_after_entity_is_fatal():
  graph = pr.ProcedureGraph(ENTITIES)
  lst = list_screen()
  graph.record(pr.make_hop(0, lst, lst.elements[0], detail("Pesto"), ENTITIES, True))
  junk = el(cls="android.widget.ImageView", bounds=(0, 0, 50, 50), clickable=True)
  graph.record(pr.make_hop(1, screen(DETAIL, junk), junk, detail("Pesto"), ENTITIES, True))
  dlg = screen(DIALOG, el(text="DELETE", rid="android:id/button1", cls="android.widget.Button",
                          bounds=(700, 900, 900, 1000), clickable=True))
  graph.record(pr.make_hop(2, dlg, dlg.elements[0], lst, ENTITIES, False))
  assert graph.template() is None


def _form_demo(graph):
  """Item 1 of an add-multiple task: +, name, amount, category, note | Save."""
  fab = el(desc="Add", rid="app:id/fab", cls="android.widget.ImageButton", bounds=(900, 2000, 1000, 2100),
           clickable=True)
  name_f = el(rid="app:id/name", cls="android.widget.EditText", bounds=(0, 300, 1000, 400), clickable=True)
  amt_f = el(rid="app:id/amount", cls="android.widget.EditText", bounds=(0, 500, 1000, 600), clickable=True)
  note_f = el(rid="app:id/note", cls="android.widget.EditText", bounds=(0, 900, 1000, 1000), clickable=True)
  def opt(label, y):
    return [el(rid="app:id/cat", cls="android.widget.LinearLayout", bounds=(0, y, 1000, y + 80),
               clickable=True), el(text=label, cls="android.widget.TextView", bounds=(10, y + 5, 500, y + 70))]
  form = screen("app/.Form", name_f, amt_f, note_f, opt("Social", 700), opt("Others", 800))
  lst = screen(LIST, fab)
  graph.record(pr.make_hop(0, lst, fab, form, (), True))
  graph.record(pr.make_typed_hop(1, form, name_f, "Social Club Dues", form, (), True))
  graph.record(pr.make_typed_hop(2, form, amt_f, "67.41", form, (), True))
  graph.record(pr.make_hop(3, form, form.elements[3], form, (), True))   # category "Social"
  graph.record(pr.make_typed_hop(4, form, note_f, "Monthly recurring", form, (), True))
  save = el(text="SAVE", rid="app:id/save", cls="android.widget.Button", bounds=(0, 2200, 300, 2300),
            clickable=True)
  graph.record(pr.make_hop(5, screen("app/.Form", save), save, lst, (), False))
  return lst, form


def test_record_slots_replay_the_next_items_values():
  goal = ("Add the following expenses: Expense: Social Club Dues amount $67.41 category Social note "
          "Monthly recurring. Expense: Legal Fees amount $10.14 category Others note Monthly recurring.")
  out = ('{"items":[{"name":"Expense: Social Club Dues","fields":{"amount":"$67.41","category":"Social",'
         '"note":"Monthly recurring"}},{"name":"Expense: Legal Fees","fields":{"amount":"$10.14",'
         '"category":"Others","note":"Monthly recurring"}}]}')
  names, records = pr.parse_obligation_records(out, goal)
  assert names == ("Social Club Dues", "Legal Fees")
  graph = pr.ProcedureGraph(())
  lst, form = _form_demo(graph)
  assert graph.template() is None  # without a contract, typed values are content
  graph.set_entities(names)
  graph.set_records(records)
  template = graph.template()
  assert [h.field or ("ENTITY" if h.entity else "") for h in template] == [
      "", "ENTITY", "amount", "category", "note"]
  planned, element = graph.plan(lst)
  assert planned.entity == "Legal Fees" and planned.values["amount"] == "10.14"
  screens = iter([form] * 12)
  acts = []
  result = pr.execute(planned, element, capture=lambda: next(screens), click=acts.append,
                      sleep=lambda _: None)
  assert result.reason == "stopped_before_commit" and result.completed == 5
  typed = [a["text"] for a in acts if a["action_type"] == "input_text"]
  assert typed == ["Legal Fees", "10.14", "Monthly recurring"]
  pressed = [a for a in acts if a["action_type"] == "click"]
  assert pressed[-1]["y"] == 840  # the "Others" option, not "Social"


def test_record_without_needed_field_is_refused():
  graph = pr.ProcedureGraph(())
  lst, _ = _form_demo(graph)
  graph.set_entities(("Social Club Dues", "Legal Fees"))
  graph.set_records({"Social Club Dues": {"amount": "67.41", "category": "Social", "note": "Monthly recurring"},
                     "Legal Fees": {"amount": "10.14"}})
  assert graph.plan(lst) is None and graph.last_rejection == "record_missing_field"


def test_swipe_inside_a_form_is_replayed_as_the_same_swipe():
  graph = pr.ProcedureGraph(())
  lst, form = _form_demo(graph)
  # Insert a scroll between the amount and the category press of item 1.
  hops = graph.hops
  swipe = pr.make_swipe_hop(2, form, "down", form, True)
  graph.hops = hops[:3] + [swipe] + hops[3:]
  graph.set_entities(("Social Club Dues", "Legal Fees"))
  graph.set_records({"Social Club Dues": {"amount": "67.41", "category": "Social", "note": "Monthly recurring"},
                     "Legal Fees": {"amount": "10.14", "category": "Others", "note": "Monthly recurring"}})
  template = graph.template()
  assert [h.action_type for h in template].count("swipe") == 1
  planned, element = graph.plan(lst)
  acts = []
  result = pr.execute(planned, element, capture=lambda: form, click=acts.append, sleep=lambda _: None)
  assert result.reason == "stopped_before_commit"
  assert {"action_type": "swipe", "direction": "down"} in acts


def test_typed_value_keeps_the_models_punctuation():
  graph = pr.ProcedureGraph(())
  lst, form = _form_demo(graph)
  graph.hops[4] = dataclasses.replace(graph.hops[4], inner=("Monthly recurring.",))
  graph.set_entities(("Social Club Dues", "Legal Fees"))
  graph.set_records({"Social Club Dues": {"amount": "67.41", "category": "Social", "note": "Monthly recurring"},
                     "Legal Fees": {"amount": "10.14", "category": "Others", "note": "Paid monthly"}})
  planned, element = graph.plan(lst)
  acts = []
  pr.execute(planned, element, capture=lambda: form, click=acts.append, sleep=lambda _: None)
  assert [a["text"] for a in acts if a["action_type"] == "input_text"][-1] == "Paid monthly."


def test_no_new_replay_until_the_opened_entity_is_committed():
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  assert graph.plan(list_screen(ENTITIES[1:])) is not None
  graph.opened("Garlic Shrimp")  # a replay started on it, then stopped early
  graph.mark_failed("Garlic Shrimp")
  assert graph.plan(list_screen(ENTITIES[1:])) is None
  assert graph.last_rejection == "entity_in_progress"
  # A navigation hop does not release it; only an observed commit does.
  lst = list_screen(ENTITIES[1:])
  graph.record(pr.make_hop(5, lst, lst.elements[0], detail("Garlic Shrimp"), ENTITIES, True))
  assert graph.plan(list_screen(ENTITIES[1:])) is None
  dialog = screen(DIALOG, el(text="DELETE", rid="android:id/button1", cls="android.widget.Button",
                             bounds=(700, 900, 900, 1000), clickable=True))
  graph.record(pr.make_hop(6, dialog, dialog.elements[0], list_screen(("Lentil Soup",)),
                           ENTITIES, False))
  planned, _ = graph.plan(list_screen(("Lentil Soup",)))
  assert planned.entity == "Lentil Soup"


def test_naive_graph_ablation_replays_by_demonstrated_coordinates(monkeypatch):
  monkeypatch.setenv("ANDROID_WORLD_NAIVE_GRAPH", "1")
  graph = pr.ProcedureGraph(ENTITIES)
  demonstrate(graph)
  planned, element = graph.plan(list_screen(ENTITIES[1:]))
  clicks = []
  screens = iter([detail("Garlic Shrimp"), menu("Garlic Shrimp"), screen(DIALOG, el(text="DELETE"))])
  out = pr.execute(planned, element, capture=lambda: next(screens), click=clicks.append,
                   sleep=lambda s: None)
  # Every press lands where the demonstration (on Pesto, row 1) pressed.
  assert [(c["x"], c["y"]) for c in clicks] == [h.xy for h in planned.hops]
  assert out.completed <= len(planned.hops)
