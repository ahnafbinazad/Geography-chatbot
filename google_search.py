from googlesearch import search


def google(query, num_results=5):
    try:
        # Performing Google search with the specified query
        search_results = search(query, stop=num_results)

        # Printing the search results
        for i, result in enumerate(search_results, start=1):
            print(f"Result {i}: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Query to search on Google
    query = "what is the golden gate bridge made of"

    # Number of search results to retrieve (default is 5)
    num_results = 5

    # Calling the function to perform Google search
    google(query)
