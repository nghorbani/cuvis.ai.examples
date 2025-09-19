import sys

def main():
    msg = (
        "This script is deprecated.\n"
        "Use the new CLI instead:\n"
        "  cuvisai-train model=efficientad/medium dataset=efficientad_train_val\n"
        "Override params via Hydra, e.g. trainer.max_epochs=1\n"
    )
    print(msg)
    sys.exit(1)

if __name__ == "__main__":
    main()
