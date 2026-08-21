# Phase 0b device measurement — 2026-08-21

Raw data backing SPEC.md §7 / §9 Phase 0b. Measured on the actual reference device
(Oppo Reno 7 5G, CPH2371, MediaTek Dimensity 900), reached via a tailnet proxy host
with the phone already attached over USB adb.

- `throughput.jsonl` — `llama-bench` output (`-o jsonl`), 3 quants (Q8_0, Q4_0, the
  prior project's Q4_K_M fine-tune) × depth sweep (0 / ~4k / ~8k, big-cores-only) ×
  core-mask sweep (big-only `0xC0` / LITTLE-only `0x3F` / all-cores `0xFF`, depth 0),
  each `-pg 2500,150 -ub 128 -fa on -lm mmap -r 2`.
- `rss.txt` — `VmHWM` + `smaps_rollup` `Pss` per quant, one real 2,500-token completion
  at `-c 4096 --no-mmap -C 0xC0 -t 2`.

Build: NDK r27c, `arm64-v8a`, `GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod`
(matches the Reno 7's Cortex-A78: `dotprod` present, no `i8mm`/SVE).

See `SPEC.md` §7 and §9 Phase 0b for the interpreted findings and gate verdict.
