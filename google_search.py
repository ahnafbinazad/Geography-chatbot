from googlesearch import search


def google(query, num_results=5):
    try:
        search_results = search(query, num_results)

        # Printing the search results
        for i, result in enumerate(search_results, start=1):
            print(f"Result {i}: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
