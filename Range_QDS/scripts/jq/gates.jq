{
  final_claim_blocking_gates: .final_claim_summary.blocking_gates,
  workload_stability: {
    pass: .workload_stability_gate.gate_pass,
    failed: .workload_stability_gate.failed_checks
  },
  support_overlap: {
    pass: .support_overlap_gate.gate_pass,
    failed: .support_overlap_gate.failed_checks
  },
  predictability: {
    pass: .predictability_audit.gate_pass,
    checks: .predictability_audit.gate_checks,
    metrics: .predictability_audit.metrics
  },
  prior_predictive_alignment: .predictability_audit.prior_predictive_alignment_gate,
  target_diffusion: {
    pass: .target_diffusion_gate.gate_pass,
    failed: .target_diffusion_gate.failed_checks
  },
  workload_signature: .workload_distribution_comparison.workload_signature_gate,
  learning_causality: {
    pass: .learning_causality_summary.learning_causality_gate_pass,
    failed: .learning_causality_summary.learning_causality_failed_checks,
    prior_sample_gate_pass: .learning_causality_summary.prior_sample_gate_pass,
    prior_sample_gate_failures: .learning_causality_summary.prior_sample_gate_failures,
    deltas: {
      shuffled_score: .learning_causality_summary.shuffled_score_ablation_delta,
      untrained: .learning_causality_summary.untrained_score_ablation_delta,
      shuffled_prior: .learning_causality_summary.shuffled_prior_field_ablation_delta,
      no_query_prior: .learning_causality_summary.no_query_prior_field_ablation_delta,
      no_behavior_head: .learning_causality_summary.no_behavior_head_ablation_delta,
      no_segment_budget_head: .learning_causality_summary.no_segment_budget_head_ablation_delta,
      no_fairness_preallocation: .learning_causality_summary.no_trajectory_fairness_preallocation_ablation_delta
    }
  },
  global_sanity: {
    pass: .global_sanity_gate.gate_pass,
    failed: .global_sanity_gate.failed_checks,
    length: .global_sanity_gate.avg_length_preserved,
    sed_ratio: .global_sanity_gate.avg_sed_ratio_vs_uniform
  }
}
