import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from images import load_generated_images


def metric(x: np.ndarray, y: np.ndarray) -> np.float64:
    """
    A generalization of the Frobenius norm to calculate the distance between two N-dimensional arrays.

    Args:
        x (`np.ndarray`):
            One of the two arrays.
        y (`np.ndarray`):
            One of the two arrays.

    Returns:
        `np.float64`: The distance between the two arrays.
    """

    # Frobenius norm generalization
    return np.sum((x - y) ** 2) ** 0.5


def calculate_pairwise_metrics(images: list) -> np.ndarray:
    """
    Calculate distances between each pair of images in a list.

    Args:
        images (`list`):
            A list of images

    Returns:
        `np.ndarray`: An array of distances between each unique pair of images in the list.
    """

    # Calculate pairwise metrics
    metrics = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            metrics.append(metric(images[i], images[j]))
    return np.array(metrics)


def main() -> None:
    """
    Create histograms of pairwise distances between generated images.
    """

    # Prompts to use as inputs
    prompts = {
        "main": "a photo capturing student life on campus at the University of Toronto",
        "similar": "a photo capturing student life on campus at the University of Waterloo",
        "different": "a wide shot of Santa Monica Beach",
    }

    # Change plot font size
    mpl.rcParams["font.size"] = 16

    # Create a histogram for each prompt
    for prompt_name in prompts:

        # Calculate pairwise metrics for first and last generations
        metrics = []
        for generation in [0, 19]:
            dataset = load_generated_images(prompts, prompt_name, generation)
            images = [np.array(dataset[i]["image"], dtype=np.float64) for i in range(dataset.num_rows)]
            metrics.append(calculate_pairwise_metrics(images))
        
        # Create a histogram
        fig, ax = plt.subplots(layout="constrained")
        ax.hist(metrics, bins=20, density=True, histtype="stepfilled", label=["First Generation", "Last Generation"], alpha=0.5)

        # Add labels
        ax.set_title(f"Distribution of pairwise distance metrics\nfor prompt {prompt_name}")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Probability")

        # Add legend
        ax.legend(loc="upper left", reverse=True, framealpha=0.3)

        # Set y-axis view limits
        ax.set_ylim(top=10 ** -4)

        # Format tick values
        ax.ticklabel_format(scilimits=[-4, 5], useMathText=True)

        # Display plot
        plt.show()


if __name__ == "__main__":
    main()
