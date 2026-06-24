# CoT Faithfulness

This experiment asks a simple question: when an open-source model shows its reasoning, does that reasoning actually support the answer it ends up giving?

The setup uses 50 ARC-Challenge science questions, three generator models, and one fixed judge model. The generator answers with chain-of-thought plus a final multiple-choice answer. The judge only sees the reasoning trace and predicts which answer that reasoning implies. If the implied answer matches the model's actual answer, the trace is treated as faithful.

## Included Here

- `data/`: sampled ARC dataset and prep script
- `results/`: saved JSON outputs for each model run
- `plots/`: exported charts and visual assets
- `final_visuals/`: final selected graphics
- `visuals/`: scripts that generate the presentation visuals
- `style/`: shared plotting styles and palette
- `generate_and_judge.py`: main pipeline for generation and judging
- `evaluate.py`: metrics and plot generation

## Models

- Generators: `llama3.1:8b`, `mistral:7b`, `deepseek-r1:7b`
- Judge: `qwen3:14b`

## Quick Start

```bash
cd cot-faithfulness
pip install -r requirements.txt
ollama pull qwen3:14b
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull deepseek-r1:7b
python generate_and_judge.py --model llama3.1:8b
python generate_and_judge.py --model mistral:7b
python generate_and_judge.py --model deepseek-r1:7b
python evaluate.py
```

## Current Snapshot

| Model | Valid judged rows | Faithful | Accuracy |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B | 50 | 84% | 88% |
| Mistral 7B | 50 | 84% | 70% |
| DeepSeek-R1 7B | 30 | 27% | 97% |

The short version: answer accuracy and reasoning faithfulness do not move together cleanly. Llama and Mistral show much stronger visible alignment between reasoning and answer than DeepSeek-R1 in this setup, while DeepSeek-R1 stays highly accurate on the smaller subset of judgeable traces.

## Notes

- The sample is small, so treat the results as exploratory.
- DeepSeek-R1 produced many ambiguous traces in this pipeline, which makes its faithfulness estimate less stable.
- `RESULTS_INTERPRETATION.md` contains the fuller readout.
- `ANCHOR_PLAN.md` captures the original experiment plan and framing.
