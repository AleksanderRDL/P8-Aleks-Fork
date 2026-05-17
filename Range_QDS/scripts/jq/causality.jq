{
  learning_causality_status: .learning_causality_summary.learning_causality_ablation_status,
  learning_causality_gate_pass: .learning_causality_summary.learning_causality_gate_pass,
  failed_checks: .learning_causality_summary.learning_causality_failed_checks,
  selector_control: {
    learned_controlled_retained_slot_fraction: .learning_causality_summary.learned_controlled_retained_slot_fraction,
    planned_learned_controlled_retained_slot_fraction: .learning_causality_summary.planned_learned_controlled_retained_slot_fraction,
    actual_learned_controlled_retained_slot_fraction: .learning_causality_summary.actual_learned_controlled_retained_slot_fraction,
    trajectories_with_at_least_one_learned_decision: .learning_causality_summary.trajectories_with_at_least_one_learned_decision,
    segment_budget_entropy_normalized: .learning_causality_summary.segment_budget_entropy_normalized
  },
  deltas: {
    shuffled_score: .learning_causality_summary.shuffled_score_ablation_delta,
    untrained: .learning_causality_summary.untrained_score_ablation_delta,
    shuffled_prior: .learning_causality_summary.shuffled_prior_field_ablation_delta,
    prior_field_only: .learning_causality_summary.prior_field_only_score_ablation_delta,
    no_query_prior: .learning_causality_summary.no_query_prior_field_ablation_delta,
    no_behavior_head: .learning_causality_summary.no_behavior_head_ablation_delta,
    no_segment_budget_head: .learning_causality_summary.no_segment_budget_head_ablation_delta,
    no_fairness_preallocation: .learning_causality_summary.no_trajectory_fairness_preallocation_ablation_delta,
    no_geometry_tie_breaker: .learning_causality_summary.no_geometry_tie_breaker_ablation_delta
  },
  prior_sample_gate: {
    pass: .learning_causality_summary.prior_sample_gate_pass,
    failures: .learning_causality_summary.prior_sample_gate_failures,
    shuffled_prior: .learning_causality_summary.prior_sensitivity_diagnostics.shuffled_prior_fields.sampled_prior_features,
    no_prior: .learning_causality_summary.prior_sensitivity_diagnostics.without_query_prior_features.sampled_prior_features
  },
  selector_config: .learning_causality_summary.learned_segment_selector_config
}
