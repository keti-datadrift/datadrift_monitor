"""Retired entrypoint kept only to make old commands fail safely."""


def main() -> None:
    raise SystemExit(
        "scripts/train_pycil.py is retired: reference repositories are now "
        "read-only algorithm references. Use `python scripts/train_table1.py "
        "configs/table1/cifar100/<method>.yaml`."
    )


if __name__ == "__main__":
    main()
