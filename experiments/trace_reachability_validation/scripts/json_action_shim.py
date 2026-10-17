"""Tiny helper so scripts build android_world JSONActions uniformly."""
from android_world.env import json_action


def click(x, y):
  return json_action.JSONAction(action_type="click", x=int(x), y=int(y))
