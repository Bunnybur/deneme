"""
Pipeline Execution Script
=========================
Runs the complete data processing pipeline in order.

Steps:
1. Data cleaning (drop SensorId, handle nulls)
2. Data standardization (StandardScaler)
3. Data analysis (visualizations)
4-6. Model training (Supervised ML + Isolation Forest + Autoencoder)

Usage: python run_pipeline.py [--skip-viz]
"""

import subprocess
import sys
import os

def run_step(step_name, script_path, skip=False):
    """Run a pipeline step."""
    if skip:
        print(f"\n⏭️  Skipping: {step_name}")
        return True
    
    print(f"\n{'='*70}")
    print(f"Running: {step_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error in {step_name}")
        print(f"Exit code: {e.returncode}")
        return False

def main():
    """Run the complete pipeline."""
    skip_viz = '--skip-viz' in sys.argv
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        SENSOR FAULT DETECTION - DATA PIPELINE                    ║
║        Advanced Computer Programming Course                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Pipeline steps
    steps = [
        ("Step 1: Data Cleaning", "src/1_clean_data.py", False),
        ("Step 2: Data Standardization", "src/2_standardize_dataset.py", False),
        ("Step 3: Data Analysis", "src/3_analyze_data.py", skip_viz),
        ("Step 4: Train Supervised Models", "src/4_train_supervised_models.py", False),
        ("Step 5: Train Isolation Forest", "src/5_train_isolation_forest.py", False),
        ("Step 6: Train Autoencoder", "src/6_train_autoencoder.py", False),
    ]
    
    # Run each step
    for step_name, script_path, skip in steps:
        success = run_step(step_name, script_path, skip)
        if not success:
            print(f"\n❌ Pipeline failed at: {step_name}")
            sys.exit(1)
    
    # Success message
    print(f"\n{'='*70}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'='*70}\n")
    print("✅ All 6 steps completed successfully")
    print("\nModels Trained:")
    print("  → Supervised: Random Forest, Gradient Boosting, XGBoost, Logistic Regression")
    print("  → Unsupervised: Isolation Forest, Autoencoder")
    print("\nNext Steps:")
    print("  → Run 'python src/main.py' to start the API server")
    print("  → Or run 'uvicorn src.main:app --reload' for development")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
