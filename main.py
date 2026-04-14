# main.py
# Entry point — runs the complete threat detection pipeline

from src.load_data    import load_nslkdd
from src.preprocess   import preprocess
from src.train_model  import train_and_evaluate
from src.visualize    import (plot_label_distribution, plot_confusion_matrix,
                               plot_feature_importance, plot_anomaly_scores)

if __name__ == "__main__":

    # ── Step 1: Load Data ──────────────────────────────────────────
    print("=" * 50)
    print("STEP 1: Loading Dataset")
    print("=" * 50)
    train_df, test_df = load_nslkdd()

    # ── Step 2: Preprocess ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 2: Preprocessing")
    print("=" * 50)
    X_train, X_test, y_train, y_test, feature_names = preprocess(train_df, test_df)

    # ── Step 3: Visualize label distribution ──────────────────────
    plot_label_distribution(y_train)

    # ── Step 4: Train and Evaluate ─────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 3: Training Model")
    print("=" * 50)
    model, y_pred, cm = train_and_evaluate(X_train, X_test, y_train, y_test)

    # ── Step 5: Generate all graphs ───────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 4: Generating Visualizations")
    print("=" * 50)
    plot_confusion_matrix(cm)
    plot_feature_importance(model, feature_names)
    plot_anomaly_scores(model, X_test, y_test)

    print("\n Pipeline complete. Check images/ folder for graphs.")