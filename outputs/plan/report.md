# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: live
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.95 | 0.96 | 0.01 |
| Avg attempts | 1 | 1.12 | 0.12 |
| Avg token estimate | 384.92 | 797.37 | 412.45 |
| Avg latency (ms) | 1598.94 | 3933.91 | 2334.97 |

## Failure modes
```json
{
  "react": {
    "none": 95,
    "wrong_final_answer": 4,
    "entity_drift": 1
  },
  "reflexion": {
    "none": 96,
    "wrong_final_answer": 2,
    "incomplete_multi_hop": 1,
    "entity_drift": 1
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding
- plan_then_execute

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
