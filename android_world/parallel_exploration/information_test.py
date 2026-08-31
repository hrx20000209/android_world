

def test_reasoning_prior_reads_what_the_model_said_it_wants():
  """Eq.(4): K_target / A_expect / K_risk / U_miss from the model's own text."""
  from android_world.parallel_exploration.information import parse_reasoning_prior
  output = (
      "<THINK> I see the list of files in Markor. The task is to delete the "
      "note named 'bold_king_edited'. I need to long-press the file to reveal "
      "the delete option in the menu. </THINK>\n"
      "explain:I need to long-press the 'bold_king_edited' note to select it"
  )
  prior = parse_reasoning_prior(output, "some unrelated global goal")
  assert prior.source == "reasoning_prior"
  assert "bold_king_edited" in prior.required_information_slots
  assert "menu" in prior.expected_affordances
  assert "delete" in prior.risk_constraints
  # The subgoal is this step's intent, not the episode-wide task text.
  assert "long-press" in prior.current_subgoal
  assert "unrelated global goal" not in prior.current_subgoal


def test_reasoning_prior_falls_back_when_there_is_no_reasoning_text():
  """A model emitting a bare tool call still has to work, just without P."""
  from android_world.parallel_exploration.information import parse_reasoning_prior
  bare = '<tool_call>{"name":"mobile_use","arguments":{"action":"click"}}</tool_call>'
  prior = parse_reasoning_prior(bare, "find contact")
  assert prior.source != "reasoning_prior"
  assert prior.current_subgoal == "find contact"


def test_reasoning_prior_distinguishes_consecutive_steps():
  """The point of P: the task goal is constant, the prior is not."""
  from android_world.parallel_exploration.information import parse_reasoning_prior
  goal = "Delete the note named bold_king_edited in Markor"
  step1 = parse_reasoning_prior("<THINK>I should open the Markor app first.</THINK>", goal)
  step2 = parse_reasoning_prior(
      "<THINK>Markor is open. I need to find the search field.</THINK>", goal)
  assert step1.current_subgoal != step2.current_subgoal
  assert "search" in step2.expected_affordances
  assert "search" not in step1.expected_affordances
