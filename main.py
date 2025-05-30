from experiments import (
    run_batch_size_experiment,
    run_optimizer_comparison,
    run_regularization_experiment,
    run_learning_rate_schedule_experiment,
    run_small_batch_size_experiment
)
from visualization import (
    plot_batch_size_results,
    plot_optimizer_comparison,
    plot_learning_curves,
    plot_regularization_results
)


def main():
    print("Running small batch size experiment...")
    small_batch_size_results = run_small_batch_size_experiment()
    small_batch_size_results.to_csv('small_batch_size_results.csv', index=False)
    plot_batch_size_results(small_batch_size_results)


    print("Running batch size experiment...")
    batch_size_results = run_batch_size_experiment()
    batch_size_results.to_csv('batch_size_results.csv', index=False)
    plot_batch_size_results(batch_size_results)

    print("\nRunning optimizer comparison experiment...")
    optimizer_results = run_optimizer_comparison()
    optimizer_results.to_csv('optimizer_results.csv', index=False)
    plot_optimizer_comparison(optimizer_results)
    plot_learning_curves(optimizer_results, 'optimizer')

    print("\nRunning regularization experiment...")
    regularization_results = run_regularization_experiment()
    regularization_results.to_csv('regularization_results.csv', index=False)
    plot_regularization_results(regularization_results)

    print("\nRunning learning rate schedule experiment...")
    lr_schedule_results = run_learning_rate_schedule_experiment()
    lr_schedule_results.to_csv('lr_schedule_results.csv', index=False)
    plot_learning_curves(lr_schedule_results, 'schedule')

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()