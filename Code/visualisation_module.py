import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_adaptive_optimization_progress(export_folder: str):
    """
    Visualise the evolution of alfa weights and total underqualified assignments during adaptive optimization.
    Exports two plots:
      - One showing the change in alfa weights over iterations (stacked area plot)
      - One showing total underqualified count over iterations (line plot)
    """
    history_path = os.path.join(export_folder, "optimization_history.xlsx")

    if not os.path.exists(history_path):
        raise FileNotFoundError(f"❌ Bestand niet gevonden: {history_path}")

    df = pd.read_excel(history_path)

    # === Plot 1: Alfa-gewichten doorheen iteraties ===
    alfas = pd.DataFrame(df["alfa_p"].apply(eval).to_list())
    alfas.index = df["iteration"]

    plt.figure(figsize=(12, 6))
    alfas.plot.area()
    plt.title("📈 Evolutie van alfa-gewichten per positie")
    plt.xlabel("Iteratie")
    plt.ylabel("Alfa-gewicht")
    plt.legend(title="Positie", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(export_folder, "plot_alfa_weights.png"))
    plt.close()

    # === Plot 2: Underqualified count doorheen iteraties ===
    plt.figure(figsize=(8, 5))
    plt.plot(df["iteration"], df["total"], marker="o")
    plt.title("📉 Totaal aantal underqualified assignments")
    plt.xlabel("Iteratie")
    plt.ylabel("Aantal underqualified toewijzingen")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(export_folder, "plot_underqualified_total.png"))
    plt.close()

    print("✅ Visualisaties opgeslagen in:", export_folder)
