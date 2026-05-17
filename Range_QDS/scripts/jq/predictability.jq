{
  gate_pass: .predictability_audit.gate_pass,
  gate_checks: .predictability_audit.gate_checks,
  metrics: .predictability_audit.metrics,
  prior_predictive_alignment_gate: .predictability_audit.prior_predictive_alignment_gate,
  per_head_predictability: .predictability_audit.per_head_predictability,
  prior_channel_predictability: .predictability_audit.prior_channel_predictability,
  support_overlap_gate: {
    pass: .support_overlap_gate.gate_pass,
    sampled_prior_nonzero_fraction: .support_overlap_gate.sampled_prior_nonzero_fraction,
    primary_sampled_prior_nonzero_fraction: .support_overlap_gate.primary_sampled_prior_nonzero_fraction,
    query_prior_support_overlap: .support_overlap_gate.query_prior_support_overlap,
    route_density_overlap: .support_overlap_gate.route_density_overlap,
    eval_points_outside_train_prior_extent_fraction: .support_overlap_gate.eval_points_outside_train_prior_extent_fraction
  }
}
