if __name__ == "__main__":
    # Seed
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    EPOCHS = 20
    batch_size = 2048
    noise_levels = [0.0, 0.2, 0.3, 0.4, 0.5]  # Different noise levels (0%, 20%, ..., 50%)

    # Results dictionary to store accuracies
    results = {
        "PC-Only": {},
        "Hebb-Only": {},
        "Hybrid": {}
    }

    for noise_std in noise_levels:
        print(f"\n### Training with Noise Level: {noise_std * 100:.0f}% ###")

        # Update transforms for the current noise level
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten
            transforms.Lambda(lambda x: add_noise_to_image(x, noise_std=noise_std))  # Add noise
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])

        # Load data
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Train and test PC-Only
        print("\n================= PC-Only =================")
        accs_pc_only = run_experiment_pc_only(
            train_loader, test_dataset, device,
            epochs=EPOCHS,
            T=20,
            lr_x=0.01,
            lr_p=0.001,
            batch_size=batch_size
        )
        results["PC-Only"][noise_std] = accs_pc_only

        # Train and test Hebb-Only
        print("\n================= Hebb-Only =================")
        accs_hebb_only = run_experiment_hebb_only(
            train_loader, test_dataset, device,
            epochs=EPOCHS,
            lr=0.001,
            batch_size=batch_size
        )
        results["Hebb-Only"][noise_std] = accs_hebb_only

        # Train and test Hybrid
        print("\n================= Hybrid =================")
        accs_hybrid = run_experiment_hybrid(
            train_loader, test_dataset, device,
            epochs=EPOCHS,
            T=20,
            lr_x=0.01,
            lr_p=0.001,
            batch_size=batch_size
        )
        results["Hybrid"][noise_std] = accs_hybrid

    # Plot results for all configurations and noise levels
    plt.figure(figsize=(10, 6))
    for config, config_results in results.items():
        for noise_std, accs in config_results.items():
            plt.plot(range(EPOCHS + 1), accs, label=f"{config} - Noise {noise_std * 100:.0f}%")
    plt.title("MNIST: Configurations with Varying Noise Levels")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save results to a text file
    output_file = "results_summary.txt"
    with open(output_file, "w") as f:
        f.write("MNIST: Configurations with Varying Noise Levels\n")
        f.write("="*50 + "\n")
        for config, config_results in results.items():
            f.write(f"\n### {config} ###\n")
            for noise_std, accs in config_results.items():
                f.write(f"Noise Level {noise_std * 100:.0f}%:\n")
                f.write(", ".join(f"{acc:.4f}" for acc in accs) + "\n")
        f.write("\nResults saved successfully!\n")
    
    print(f"Results have been saved to {output_file}.")

