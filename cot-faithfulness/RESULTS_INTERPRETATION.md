# CoT Faithfulness Mini Experiment

## Setup

- Dataset: ARC-Challenge sample
- Questions per generator: 50
- Generator models: Llama 3.1 8B, Mistral 7B, DeepSeek-R1 7B
- Judge model: Qwen3 14B
- Judging rule: the judge sees only the model's chain-of-thought and predicts the final answer implied by that reasoning

## Core Results

| Model | Valid judged rows | Faithful | Accuracy | Faithful when correct | Faithful when incorrect | Avg CoT words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3.1 8B | 50 | 84% | 88% | 84% | 83% | 212.8 |
| Mistral 7B | 50 | 84% | 70% | 91% | 67% | 90.5 |
| DeepSeek-R1 7B | 30 | 27% | 97% | 28% | 0% | 34.2 |

## Outcome Counts

| Model | Faithful + Correct | Faithful + Incorrect | Unfaithful | Ambiguous |
| --- | ---: | ---: | ---: | ---: |
| Llama 3.1 8B | 37 | 5 | 3 | 5 |
| Mistral 7B | 32 | 10 | 1 | 7 |
| DeepSeek-R1 7B | 8 | 0 | 7 | 35 |

## Interpretation

Llama 3.1 8B and Mistral 7B tie on overall faithfulness at 84%, but they get there differently. Llama is the stronger all-around model here because it pairs that faithfulness with much higher answer accuracy, 88% versus 70%.

Mistral is the cleanest example of reasoning that tracks the final answer when the model is right. Its faithfulness rises to 91% on correct answers, which suggests its visible chain-of-thought is often genuinely aligned with the answer it produces.

DeepSeek-R1 7B behaves very differently. On the subset of rows that yielded valid judged traces, it is highly accurate, 29 correct out of 30. But its faithfulness is much lower at 27%, and it produces 35 ambiguous rows. In this setup, that means it often either does not expose reasoning in a way the judge can reliably map to the answer, or its visible rationale does not actually support the answer it gives.

The scatter plot also argues against a simple "longer reasoning means better answers" story. Llama produces the longest chains-of-thought and performs well, but Mistral is much shorter while matching Llama on faithfulness. DeepSeek is shortest and least faithful.

## Main Takeaway

This experiment suggests that answer accuracy and reasoning faithfulness are separable. A model can be accurate without exposing a chain-of-thought that clearly supports its answer, and a model can be highly faithful without being the most accurate overall.

## Caveats

- Small sample size: 50 questions per generator
- DeepSeek's metrics are based on only 30 valid judged rows, so its accuracy and faithfulness are less stable
- The judge is a separate model and reflects this particular prompting and extraction setup
- "Ambiguous" means the judged reasoning trace did not cleanly imply A, B, C, or D

## Visual Assets

- [plots/outcome_breakdown.png](/Volumes/Projects/Mini%20experiments/plots/outcome_breakdown.png)
- [plots/faithfulness_by_correctness.png](/Volumes/Projects/Mini%20experiments/plots/faithfulness_by_correctness.png)
- [plots/cot_length_scatter.png](/Volumes/Projects/Mini%20experiments/plots/cot_length_scatter.png)
- [plots/methodology_diagram.png](/Volumes/Projects/Mini%20experiments/plots/methodology_diagram.png)
- [plots/results_card.png](/Volumes/Projects/Mini%20experiments/plots/results_card.png)
- [plots/thumbnail.png](/Volumes/Projects/Mini%20experiments/plots/thumbnail.png)
