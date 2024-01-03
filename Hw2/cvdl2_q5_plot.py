import matplotlib.pyplot as plt


if __name__ == "__main__":


    # Plot and save the accuracy values
    bar_width = 0.35

    # Calculate average accuracy values
    avg_accuracy_with_erasing = 90.7
    avg_accuracy_without_erasing = 90.1

    plt.figure(figsize=(10, 5))

    # Plotting average accuracy values
    plt.bar(
        ["With Random Erasing", "Without Random Erasing"],
        [avg_accuracy_with_erasing, avg_accuracy_without_erasing],
        color=["blue", "orange"],
    )

    plt.xlabel("Data Augmentation")
    plt.ylabel("Average Validation Accuracy")
    plt.title("Average Validation Accuracy with and without Random Erasing")
    plt.savefig("Comparison.png")
    plt.show()
