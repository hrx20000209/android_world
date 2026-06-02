import unittest

from android_world.agents import aggressive_t2_shortcut_utils as utils


SEARCH_STATE = {
    "package": "test",
    "activity": "SearchActivity",
    "screen_role": "search",
    "anchor_labels": ["Search", "Clear query"],
    "visible_targets": ["Submit query"],
    "ui_elements": [{"resource_id": "search_src_text", "hint_text": "Search", "class_name": "EditText"}],
}


class AggressiveT2ShortcutUtilsTest(unittest.TestCase):
    def test_joplin_recipe_shortcut(self):
        plan = utils.build_shortcut_plan_for_test(
            task="What quantity of spirulina do I need for Chicken Alfredo?",
            planned_action={"action_type": "click", "label": "Search"},
            explored_t1_state={**SEARCH_STATE, "package": "net.cozic.joplin"},
        )
        self.assertEqual(plan.get("no_plan_reason"), "")
        self.assertEqual(plan["shortcut_t2_action"]["action_type"], "type")
        self.assertEqual(plan["shortcut_t2_action"]["value"], "Chicken Alfredo")
        self.assertTrue(plan["shortcut_t2_action"]["is_safe_search_input"])

    def test_files_task_shortcut(self):
        plan = utils.build_shortcut_plan_for_test(
            task="Open file task.html in Downloads",
            planned_action={"action_type": "click", "label": "Search"},
            explored_t1_state={**SEARCH_STATE, "package": "com.google.android.documentsui"},
        )
        self.assertEqual(plan.get("no_plan_reason"), "")
        self.assertEqual(plan["shortcut_t2_action"]["value"], "task.html")
        self.assertTrue(plan["shortcut_t2_action"]["is_safe_search_input"])

    def test_opentracks_duration_shortcut(self):
        plan = utils.build_shortcut_plan_for_test(
            task="How long was my skiing activity October 12 2023?",
            planned_action={"action_type": "click", "label": "Search"},
            explored_t1_state={**SEARCH_STATE, "package": "de.dennisguse.opentracks"},
        )
        self.assertEqual(plan.get("no_plan_reason"), "")
        self.assertIn(plan["shortcut_t2_action"]["value"], {"October 12 2023", "skiing"})
        self.assertEqual(utils.classify_evidence("Search", plan["task_id"]), "PARTIAL_PROGRESS")

    def test_unsafe_form_input_blocked(self):
        plan = utils.build_shortcut_plan_for_test(
            task="Add contact Ava Smith with phone number 555-0101",
            planned_action={"action_type": "click", "label": "Name"},
            explored_t1_state={**SEARCH_STATE, "package": "com.android.contacts"},
        )
        self.assertEqual(plan.get("no_plan_reason"), "unsafe_form_input")

    def test_delete_blocked(self):
        plan = utils.build_shortcut_plan_for_test(
            task="Delete note grocery list",
            planned_action={"action_type": "long_press", "label": "grocery list"},
            explored_t1_state={**SEARCH_STATE, "package": "net.gsantner.markor"},
        )
        self.assertEqual(plan.get("no_plan_reason"), "risky_action")


if __name__ == "__main__":
    unittest.main()
