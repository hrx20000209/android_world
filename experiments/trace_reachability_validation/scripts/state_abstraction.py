"""Variance-aware page identity learned from many observations of an activity (agent visits + exploration probes).

key(ui) = (package, activity, mode, core)
  mode : page / dialog / ime / search / drawer  - overlays are different states, never merged with the page below
  core : named (resource-id) skeleton items of the page, minus system bars and minus the activity's DYNAMIC items
         (items whose presence varies across observed captures of the activity: frequency in [LO, HI]).
Dynamic items are estimated from the graph's captures; exploration supplies most of the variants, so the more the
app is explored the better the page identity separates dynamic content from state.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

LO, HI, MIN_CAPS = 0.1, 0.9, 3
SYSTEM = {"statusBarBackground", "navigationBarBackground"}
DIALOG = {"buttonPanel", "parentPanel", "alertTitle", "title_template", "customPanel", "contentPanel", "topPanel"}


def mode(ui):
  app = [n for n in ui.app_nodes() if n.visible]
  rids = {n.short_rid for n in app}
  if ui.ime_visible():
    return "ime"
  if rids & DIALOG or len({n.window for n in app}) > 1:
    return "dialog"
  if "search_src_text" in rids:
    return "search"
  if any("drawer" in r.lower() or "navigation_view" in r.lower() for r in rids):
    return "drawer"
  return "page"


def named_items(ui):
  out = set()
  for item in ui.skeleton():
    parts = item.split("|")
    if len(parts) > 1 and parts[1] and parts[1] not in SYSTEM:
      out.add(item)
  return out


class Abstraction:
  def __init__(self):
    self.freq = defaultdict(Counter)       # (activity, mode) -> item counts
    self.n = Counter()

  def observe(self, ui):
    k = (ui.activity, mode(ui))
    self.n[k] += 1
    self.freq[k].update(named_items(ui))

  def dynamic(self, activity, md):
    k = (activity, md)
    if self.n[k] < MIN_CAPS:
      return set()
    return {it for it, c in self.freq[k].items() if LO <= c / self.n[k] <= HI}

  def key(self, ui):
    md = mode(ui)
    core = sorted(named_items(ui) - self.dynamic(ui.activity, md))
    h = hashlib.sha1("\n".join(core).encode()).hexdigest()[:12]
    return (ui.package, ui.activity, md, h)
