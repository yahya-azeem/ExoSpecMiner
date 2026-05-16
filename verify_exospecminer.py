import os
from exospecminer.verification import run_automated_benchmarks, run_wasp39b_case_study

if __name__ == "__main__":
    print("="*60)
    print("   EXOSPECMINER FRAMEWORK VERIFICATION SUITE   ")
    print("="*60)

    # 1. Run Automated Benchmarks
    print("\n--- 1. Running Automated Benchmarks ---")
    benchmark_results = run_automated_benchmarks(num_test_cases=50, output_path="exospecminer_benchmark_report.md")

    # 2. Run WASP-39b Case Study
    print("\n--- 2. Running WASP-39b Case Study ---")
    case_study_results = run_wasp39b_case_study(output_dir="wasp39b_verification_study", num_samples=5000)

    print("\n" + "="*60)
    print(" ALL VERIFICATION SUITES COMPLETED SUCCESSFULLY ")
    print(f" Benchmark Report: {benchmark_results['report_path']}")
    print(f" Case Study Report: {case_study_results['report_path']}")
    print("="*60)
