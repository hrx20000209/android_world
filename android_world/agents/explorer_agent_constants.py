"""Constants and prompts for ExplorerElementAgent."""

SYSTEM_PROMPT = """You are an Android GUI agent.
You receive the task, history summaries, explorer hints, and current screenshot.
Return ONLY one action in <tool_call></tool_call> with JSON.

Format:
<tool_call>
{"name":"mobile_use","arguments":{...}}
</tool_call>

Action space:
{"action":"click","coordinate":[x,y]}
{"action":"long_press","coordinate":[x,y]}
{"action":"type","text":"...","coordinate":[x,y]}
{"action":"swipe","direction":"up|down|left|right"}
{"action":"swipe","start_coordinate":[x1,y1],"end_coordinate":[x2,y2]}
{"action":"open_app","text":"app_name"}
{"action":"system_button","button":"back|home|enter"}
{"action":"answer","text":"..."}
{"action":"terminate","status":"success|fail"}
Compatibility fallback:
{"action":"click","element_id":int}
{"action":"long_press","element_id":int}
{"action":"type","element_id":int,"text":"..."}

Rules:
- Prefer coordinate-based actions. Coordinates are absolute logical pixels in the screenshot space.
- If coordinate is unavailable, you may use element_id from UI list.
- avoid repeating same no-effect action.
- For task completion, prefer {"action":"terminate","status":"success"}.
  If you output {"action":"answer", ...}, it will be treated as task completion.
""".strip()

_DIR_SET = {"up", "down", "left", "right"}
_STOP_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "then",
    "task",
    "screen",
    "button",
    "open",
    "app",
}
_RISKY_KEYWORDS = {
    "stop",
    "finish",
    "delete",
    "remove",
    "clear",
    "save",
    "warning",
    "ok",
    "details",
    "back",
    "home",
    "navigate up",
}
_LOW_VALUE_CLASS_HINTS = {
    "switch",
    "checkbox",
    "radiobutton",
    "togglebutton",
    "seekbar",
    "progressbar",
}
_LOW_VALUE_TEXT_HINTS = {
    "on",
    "off",
    "enabled",
    "disabled",
    "status",
    "state",
    "timer",
    "duration",
    "elapsed",
    "progress",
}
_NAV_HELPFUL_KEYWORDS = {
    "menu",
    "more",
    "options",
    "setting",
    "settings",
    "detail",
    "details",
    "open",
    "next",
    "continue",
    "manage",
    "list",
    "folder",
    "recordings",
    "history",
    "search",
    "add",
    "new",
    "create",
    "choose",
    "select",
    "view",
    "all",
}
_SUBMIT_ACTION_KEYWORDS = {
    "ok",
    "done",
    "confirm",
    "save",
    "next",
    "search",
    "go",
    "enter",
    "apply",
    "finish",
    "确定",
    "完成",
    "保存",
    "继续",
    "下一步",
}
_DISMISS_ACTION_KEYWORDS = {
    "cancel",
    "close",
    "back",
    "dismiss",
    "later",
    "not now",
    "no thanks",
    "skip",
    "关闭",
    "取消",
    "返回",
    "跳过",
}
_KEYBOARD_HINT_TOKENS = {
    "keyboard",
    "inputmethod",
    "latinime",
    "gboard",
    "ime",
    "key_pos",
    "keyboardview",
}
_DIALOG_HINT_TOKENS = {
    "dialog",
    "popup",
    "alert",
    "bottomsheet",
    "sheet",
    "chooser",
}
_CHOICE_HINT_TOKENS = {
    "checkbox",
    "radiobutton",
    "switch",
    "toggle",
    "spinner",
    "option",
}
_TASK_INPUT_KEYWORDS = {
    "input",
    "type",
    "enter",
    "write",
    "search",
    "text",
    "message",
    "query",
    "keyword",
    "name",
    "rename",
    "填写",
    "输入",
}
_TASK_SELECT_KEYWORDS = {
    "select",
    "choose",
    "pick",
    "option",
    "toggle",
    "switch",
    "enable",
    "disable",
    "check",
    "checkbox",
    "radio",
    "选择",
    "多选",
}
_INFO_ONLY_TEXT_HINTS = {
    "duration",
    "elapsed",
    "status",
    "state",
    "remaining",
    "kb",
    "mb",
    "gb",
    "sec",
    "min",
    "hour",
}
