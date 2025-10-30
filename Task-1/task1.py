from rapidfuzz import fuzz, process

names = [
    "Geetha", "Gita", "Githa", "Geeta", "Gitu", "Sita", "Seetha", "Sunita", "Anita",
    "Rekha", "Ritika", "Ritikaa", "Pooja", "Puja", "Prachi", "Preeti",
    "Prithi", "Pari", "Priti", "Priya", "Pria", "Divya", "Diya", "Dia",
    "Sneha", "Snehaa", "Shreya", "Shriya", "Shruti", "Shruthi", "Ankita", "Ankitha"
]

def find_similar_names(query, names_list, limit=5):

    name_map = {n.lower(): n for n in names_list}
    lower_names = list(name_map.keys())

    results = process.extract(
        query.lower(), lower_names, scorer=fuzz.ratio, limit=limit
    )

    formatted_results = [(name_map[name], score) for name, score, _ in results]

    best_match = formatted_results[0]
    return best_match, formatted_results


if __name__ == "__main__":
    print("=== Name Matching System ===")
    user_input = input("Enter a name: ").strip()

    if not user_input:
        print("Please enter a valid name.")
    else:
        best, matches = find_similar_names(user_input, names)
        print(f"\nBest Match: {best[0]} (Similarity: {best[1]:.2f}%)")
        print("\nTop Similar Names:")
        for match, score in matches:
            print(f"  {match} — {score:.2f}%")
