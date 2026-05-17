def delta($a; $b):
  if ($a | type) == "number" and ($b | type) == "number" then $a - $b else null end;

{
  final_claim_summary,
  scores: {
    mlqds_query_useful_v1: .matched.MLQDS.query_useful_v1_score,
    uniform_query_useful_v1: .matched.uniform.query_useful_v1_score,
    douglas_peucker_query_useful_v1: .matched.DouglasPeucker.query_useful_v1_score,
    mlqds_vs_uniform: delta(.matched.MLQDS.query_useful_v1_score; .matched.uniform.query_useful_v1_score),
    mlqds_vs_douglas_peucker: delta(.matched.MLQDS.query_useful_v1_score; .matched.DouglasPeucker.query_useful_v1_score)
  },
  gates: {
    workload_stability: .workload_stability_gate.gate_pass,
    support_overlap: .support_overlap_gate.gate_pass,
    predictability: .predictability_audit.gate_pass,
    prior_predictive_alignment: .predictability_audit.prior_predictive_alignment_gate.gate_pass,
    target_diffusion: .target_diffusion_gate.gate_pass,
    workload_signature: .workload_distribution_comparison.workload_signature_gate.all_pass,
    learning_causality: .learning_causality_summary.learning_causality_gate_pass,
    prior_sample: .learning_causality_summary.prior_sample_gate_pass,
    global_sanity: .global_sanity_gate.gate_pass
  }
}
