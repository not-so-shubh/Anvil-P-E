# Anvil · Problems & Evaluation

Problem statements and benchmark harnesses for the Anvil hackathon.

## P-04 Submission

Final adapter:

```text
bench-p04-pcam/adapters/myteam.py
```

Run:

```bash
cd bench-p04-pcam
python self_check.py --adapter adapters.myteam:Engine --quick
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 42 97 101 202 211 303 404 503 777 1009 1337 2027 2718 3141 4096 9001 9999 \
  --out results/final_20_seed_report.json
```

Approach summary:

Reliability-weighted target prediction
+ PCAM local flow-margin steering
+ guarded Hessian precision for near-attractor probes

## Layout

```
index.html              Full problem statement dashboard (open in a browser)
bench-p01-crdt/         Benchmark harness for P-01 · CRDT-Native OLTP
bench-p02-context/      Benchmark harness for P-02 · Persistent Context Engine
```

## Open the dashboard

```
open index.html
```

Four tabs. Two open tracks (P-01, P-02). Two sponsored tracks (P-03 Omium, P-04 MetaCognition).

## Running a benchmark

Each bench is pure Python. Install per-bench requirements if present, write an adapter that wraps your engine, and run the self-check.

```bash
# P-01
cd bench-p01-crdt
python self_check.py --adapter adapters.dummy:DummyAdapter --fk-policy cascade

# P-02
cd bench-p02-context
python self_check.py --adapter adapters.dummy:DummyAdapter
```

The dummy adapters are intentionally weak baselines — they validate the harness and demonstrate the interface. Your engine plugs in as `adapters/<yourteam>.py`.

See each bench's own README for the adapter contract, scoring axes, and the three-layer anti-gaming model (L1 canonical / L2 property-based / L3 held-out adversarial).

## License

TBD.
