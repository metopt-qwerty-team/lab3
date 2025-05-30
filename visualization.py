import matplotlib.pyplot as plt
import seaborn as sns


def plot_batch_size_results(results_df):
    print(results_df)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    sns.scatterplot(data=results_df, x='batch_size', y='mse', hue='implementation')
    plt.title('MSE by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Squared Error')

    plt.subplot(2, 3, 2)
    sns.scatterplot(data=results_df, x='batch_size', y='training_time', hue='implementation')
    plt.title('Training Time by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Time (s)')

    plt.subplot(2, 3, 3)
    sns.scatterplot(data=results_df, x='batch_size', y='memory_usage', hue='implementation')
    plt.title('Memory Usage by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MB)')

    plt.subplot(2, 3, 4)
    sns.scatterplot(data=results_df, x='batch_size', y='final_loss', hue='implementation')
    plt.title('Final Loss by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Final Loss')

    plt.subplot(2, 3, 5)
    sns.scatterplot(data=results_df, x='batch_size', y='total_flops', hue='implementation')
    plt.title('Total operations by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Total flops')

    plt.tight_layout()
    plt.savefig('batch_size_results.png')
    plt.show()


def plot_optimizer_comparison(results_df):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.barplot(data=results_df, x='optimizer', y='mse')
    plt.title('MSE by Optimizer')
    plt.xlabel('Optimizer')
    plt.ylabel('Mean Squared Error')

    plt.subplot(2, 2, 2)
    sns.barplot(data=results_df, x='optimizer', y='training_time')
    plt.title('Training Time by Optimizer')
    plt.xlabel('Optimizer')
    plt.ylabel('Training Time (s)')

    plt.subplot(2, 2, 3)
    sns.barplot(data=results_df, x='optimizer', y='memory_usage')
    plt.title('Memory Usage by Optimizer')
    plt.xlabel('Optimizer')
    plt.ylabel('Memory Usage (MB)')

    plt.subplot(2, 2, 4)
    sns.barplot(data=results_df, x='optimizer', y='final_loss')
    plt.title('Final Loss by Optimizer')
    plt.xlabel('Optimizer')
    plt.ylabel('Final Loss')

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    plt.show()


def plot_learning_curves(results_df, experiment_type):
    plt.figure(figsize=(10, 6))

    if experiment_type == 'optimizer':
        for _, row in results_df.iterrows():
            plt.plot(row['test_loss_history'], label=row['optimizer'])
        plt.title('Learning Curves by Optimizer')
    elif experiment_type == 'schedule':
        for _, row in results_df.iterrows():
            plt.plot(row['loss_history'], label=row['schedule'])
        plt.title('Learning Curves by Learning Rate Schedule')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{experiment_type}_learning_curves.png')
    plt.show()


def plot_regularization_results(results_df):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.lineplot(data=results_df, x='alpha', y='mse', hue='regularization')
    plt.title('MSE by Regularization Type')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error')
    plt.xscale('log')

    plt.subplot(2, 2, 2)
    sns.lineplot(data=results_df, x='alpha', y='training_time', hue='regularization')
    plt.title('Training Time by Regularization Type')
    plt.xlabel('Alpha')
    plt.ylabel('Training Time (s)')
    plt.xscale('log')

    plt.subplot(2, 2, 3)
    sns.lineplot(data=results_df, x='alpha', y='memory_usage', hue='regularization')
    plt.title('Memory Usage by Regularization Type')
    plt.xlabel('Alpha')
    plt.ylabel('Memory Usage (KB)')
    plt.xscale('log')

    plt.subplot(2, 2, 4)
    sns.lineplot(data=results_df, x='alpha', y='final_loss', hue='regularization')
    plt.title('Final Loss by Regularization Type')
    plt.xlabel('Alpha')
    plt.ylabel('Final Loss')
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig('regularization_results.png')
    plt.show()