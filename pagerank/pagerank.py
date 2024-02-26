import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    linked_pages = corpus.get(page, set())
    linked_pages_count = len(linked_pages)

    all_pages_count = len(corpus)
    base_probability = (1 - damping_factor) / all_pages_count

    probable_pages = {
        key: base_probability + damping_factor / linked_pages_count if key in linked_pages else base_probability
        for key in corpus
    }

    return probable_pages

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    samples=[]

    samples.append(random.choice(list(corpus.keys())))

    for i in range(1, n):
        probabilities = transition_model(corpus, samples[i-1], damping_factor)
        next_sample = random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
        samples.append(next_sample)

    sample_pagerank={}

    for sample in samples:
        if sample in sample_pagerank:
            sample_pagerank[sample] += 1
        else:
            sample_pagerank[sample] = 1
    
    for key in sample_pagerank:
        sample_pagerank[key] /= n
    
    print(sample_pagerank)
    return sample_pagerank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    iterate_pagerank={
        key: 1/len(corpus)
        for key in corpus
    }

    while True:
        prev_pagerank=iterate_pagerank.copy()

        for page in corpus:
            page_rank_sum=0
            for link in corpus:
                if page in corpus[link] or not corpus[link]:
                    num_links = len(corpus[link]) if corpus[link] else len(corpus)
                    page_rank_sum += prev_pagerank[link]/num_links
            iterate_pagerank[page] = (1 - damping_factor) / len(corpus) + damping_factor * page_rank_sum
        if all(abs(prev_pagerank[page] - iterate_pagerank[page]) < 0.001 for page in iterate_pagerank):
            break
        
    return iterate_pagerank


if __name__ == "__main__":
    main()
