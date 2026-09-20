"""python -m sifess.engines  -> help"""
print(
    "SIFESS engines. Use:\n"
    "  python -m sifess.engines.train_ssl --ablation sifess_full\n"
    "  python -m sifess.engines.linear_probe --checkpoint ...\n"
    "  python -m sifess.engines.eval_ood --checkpoint ... --probe ...\n"
    "  python -m sifess.engines.run_ablations\n"
)
