import arxiv
from smolagents import tool

@tool
def search_arxiv(query: str, max_results: int = 3) -> list:
    """
    Searches arXiv for academic papers based on a query.
    
    Args:
        query: The search terms (e.g., 'Neural SDEs time series').
        max_results: The maximum number of papers to return. Defaults to 3.
        
    Returns:
        A list of dictionaries. Each dict contains 'title', 'authors', 'year', 'url', and 'abstract'.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, 
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance # Ensures accurate semantic matching
    )
    
    results = []
    for paper in client.results(search):
        # Extract authors into a comma-separated string
        authors = ", ".join([author.name for author in paper.authors])
        
        # Append as a structured dictionary rather than a raw string
        results.append({
            "title": paper.title,
            "authors": authors,
            "year": paper.published.year,
            "url": paper.entry_id,
            "abstract": paper.summary
        })
        
    return results