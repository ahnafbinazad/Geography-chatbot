import os
import class_names

def get_subdirectories(directory):
    """
    Get a list of all subdirectories in the specified directory.

    Args:
    directory (str): The directory to search.

    Returns:
    list: A list of subdirectories.
    """
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories


def main():
    # Specify the directory to search
    directory = './flags/train'

    # Get the list of subdirectories
    subdirectories = get_subdirectories(directory)

    # Sort the subdirectories list
    subdirectories.sort()

    # Sort the class_names.countries list
    class_names.countries.sort()

    # Find the differences between the lists
    sub_diff = set(subdirectories) - set(class_names.countries)
    class_diff = set(class_names.countries) - set(subdirectories)

    # Check if the lists are the same
    if sub_diff or class_diff:
        print("The lists are not identical.")
        if sub_diff:
            print("Subdirectories not in class_names.countries:", sub_diff)
        if class_diff:
            print("Class names not in subdirectories:", class_diff)
    else:
        print("The lists are identical.")

    # Check if the indexing matches
    indexing_match = True
    for index, (subdir, country) in enumerate(zip(subdirectories, class_names.countries)):
        if subdir != country:
            indexing_match = False
            print(f"The element at index {index} does not match: {subdir} (subdirectories) != {country} (class_names.countries)")
            break
    if indexing_match:
        print("The indexing matches.")


if __name__ == "__main__":
    main()
