# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: live
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.95 | 1.0 | 0.05 |
| Avg attempts | 1 | 1.05 | 0.05 |
| Avg token estimate | 384.93 | 438 | 53.07 |
| Avg latency (ms) | 1166.05 | 1234.54 | 68.49 |

## Failure modes
```json
{
  "react": {
    "none": 95,
    "wrong_final_answer": 4,
    "entity_drift": 1
  },
  "reflexion": {
    "none": 100
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
