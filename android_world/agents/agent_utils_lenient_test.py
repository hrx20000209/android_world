from android_world.agents import agent_utils


def test_extract_json_accepts_unquoted_action_key_and_flat_point():
  value = agent_utils.extract_json(
      'Click the search icon. {action_type: "click", "point":117,918}'
  )
  assert value == {"action_type": "click", "point": [117.0, 918.0]}


def test_extract_json_prefers_last_action_object():
  value = agent_utils.extract_json(
      'Example {"action_type":"wait"} final {"action_type":"click","index":3}'
  )
  assert value == {"action_type": "click", "index": 3}
