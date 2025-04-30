# first line: 1
@memory.cache
def cached_hdbscan_fit(data, min_cluster_size, min_samples=None):
    """
    Cached version of HDBSCAN fitting using the standalone library
    """
    print(f"Computing HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    start_time = time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        core_dist_n_jobs=-1,
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(data)

    print(f"Computation took {time() - start_time:.2f} seconds")
    return clusterer, labels
