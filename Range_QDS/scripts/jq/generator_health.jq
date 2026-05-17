{
  workload_stability_gate,
  train_generation: .query_generation_diagnostics.train.query_generation,
  train_acceptance: .query_generation_diagnostics.train.range_acceptance,
  eval_generation: .query_generation_diagnostics.eval.query_generation,
  eval_acceptance: .query_generation_diagnostics.eval.range_acceptance,
  selection_generation: .query_generation_diagnostics.selection.query_generation,
  selection_acceptance: .query_generation_diagnostics.selection.range_acceptance,
  workload_signature_gate: .workload_distribution_comparison.workload_signature_gate
}
